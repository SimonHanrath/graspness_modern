import os
import sys
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Set multiprocessing start method to avoid CUDA context issues with DataLoader workers
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

torch.backends.cudnn.benchmark = True 
torch.set_float32_matmul_precision("high")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.graspnet import GraspNet
from models.loss import get_loss
from dataset.graspnet_dataset import GraspNetDataset, spconv_collate_fn, load_grasp_labels, load_grasp_labels_lazy, SceneAwareSampler

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default=None)
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--log_dir', default='logs/log')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 20000]')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size to process point clouds ')
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--resume', action='store_true', default=False, help='Whether to resume from checkpoint')
parser.add_argument('--use_amp', action='store_true', default=False,
                    help='Use torch.cuda.amp for mixed-precision training')
parser.add_argument('--single_sample', action='store_true', default=False,
                    help='Overfit test: use only 1 training sample for 10 epochs')
parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers [default: 0]')
parser.add_argument('--persistent_workers', action='store_true', default=False, 
                    help='Keep workers alive between epochs (reduces memory overhead with num_workers>0)')
parser.add_argument('--lazy_grasp_labels', action='store_true', default=False,
                    help='Use lazy loading for grasp labels to reduce memory (useful with many workers)')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay for AdamW optimizer (recommended: 0.02-0.05 for transformers) [default: 0.0]')
parser.add_argument('--backbone', type=str, default='transformer', choices=['transformer', 'transformer_pretrained', 'pointnet2', 'resunet'],
                    help='Backbone architecture [default: transformer]. Use transformer_pretrained for PTv3 with Pointcept pretrained weights.')
parser.add_argument('--grad_clip', type=float, default=0.0,
                    help='Gradient clipping max norm (recommended: 1.0-5.0 for transformers, 0 to disable) [default: 0.0]')
parser.add_argument('--ptv3_pretrained_path', type=str, default=None,
                    help='Path to PTv3 pretrained weights (.pth file). If not specified, uses models/pointcept/model_best.pth')
parser.add_argument('--enable_flash', action='store_true', default=False,
                    help='Enable flash attention in PTv3 backbone (requires flash_attn package)')
parser.add_argument('--accumulation_steps', type=int, default=1,
                    help='Gradient accumulation steps (simulate larger batch with batch_size=1) [default: 1]')
parser.add_argument('--backbone_lr_scale', type=float, default=None,
                    help='Learning rate multiplier for backbone (e.g., 0.1 for pretrained). Default: 0.1 for transformer_pretrained, 1.0 otherwise')
parser.add_argument('--enable_stable_score', action='store_true', default=False,
                    help='Enable stable score prediction to penalize grasps that may cause tipping [default: False]')
parser.add_argument('--lambda_stable', type=float, default=1.0,
                    help='Weight for stable score loss term [default: 1.0]')
# DDP arguments (set automatically by torchrun, but can be overridden)
parser.add_argument('--local_rank', type=int, default=-1,
                    help='Local rank for distributed training (set by torchrun)')


cfgs = parser.parse_args()

# Set backbone_lr_scale default once (0.1 for pretrained, 1.0 otherwise)
if cfgs.backbone_lr_scale is None:
    cfgs.backbone_lr_scale = 0.1 if cfgs.backbone == 'transformer_pretrained' else 1.0

# =============================================
# Distributed Training Setup
# =============================================
def is_distributed():
    """Check if we're running in distributed mode."""
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    """Check if this is the main process (rank 0 or non-distributed)."""
    if not is_distributed():
        return True
    return dist.get_rank() == 0

def get_world_size():
    """Get the number of processes in the distributed group."""
    if not is_distributed():
        return 1
    return dist.get_world_size()

def get_rank():
    """Get the rank of the current process."""
    if not is_distributed():
        return 0
    return dist.get_rank()

def setup_distributed():
    """Initialize distributed training if running with torchrun."""
    # Check if we're running in distributed mode (torchrun sets these env vars)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize the distributed backend
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        
        return local_rank, rank, world_size
    else:
        # Not running distributed
        return 0, 0, 1

def cleanup_distributed():
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()

# Initialize distributed training
LOCAL_RANK, GLOBAL_RANK, WORLD_SIZE = setup_distributed()


EPOCH_CNT = 0
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.resume else None
if not os.path.exists(cfgs.log_dir) and is_main_process():
    os.makedirs(cfgs.log_dir)

# Only main process writes to log file
if is_main_process():
    LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
    LOG_FOUT.write(str(cfgs) + '\n')
    if is_distributed():
        LOG_FOUT.write(f'Distributed training: world_size={WORLD_SIZE}, local_rank={LOCAL_RANK}\n')
else:
    LOG_FOUT = None


def log_string(out_str):
    if is_main_process():
        LOG_FOUT.write(out_str + '\n')
        LOG_FOUT.flush()
        print(out_str)



# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def create_dataloaders():
    """Create datasets and dataloaders. Each process creates its own."""
    # Load grasp labels (use lazy loading if specified to save memory with multiple workers)
    if cfgs.lazy_grasp_labels:
        log_string("Using lazy loading for grasp labels (memory-efficient mode)")
        grasp_labels = load_grasp_labels_lazy(cfgs.dataset_root)
    else:
        log_string("Loading all grasp labels into memory (~21GB)")
        grasp_labels = load_grasp_labels(cfgs.dataset_root)

    # Stable score settings (labels auto-computed by dataset if missing)
    if cfgs.enable_stable_score:
        log_string("Stable score prediction enabled (labels will be auto-computed if missing)")

    use_rgb = (cfgs.backbone == 'transformer_pretrained')
    train_dataset = GraspNetDataset(cfgs.dataset_root, grasp_labels=grasp_labels, camera=cfgs.camera, split='train',
                                    num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                    remove_outlier=True, augment=True, load_label=True, use_rgb=use_rgb,
                                    enable_stable_score=cfgs.enable_stable_score)
    log_string('train dataset length: ' + str(len(train_dataset)))


    # For overfitting test, use only 1 sample repeated 256 times
    if cfgs.single_sample:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, [0] * 256)
        cfgs.max_epoch = 20
        log_string('Single-sample overfitting test enabled: 256x repeated, max_epoch set to 20')
    
    # Create samplers
    # SceneAwareSampler groups samples by scene to maximize collision label cache hits
    # This is much faster than random shuffle with lazy loading
    train_sampler = None
    if is_distributed():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        log_string(f'Using DistributedSampler with {WORLD_SIZE} processes')
    elif cfgs.single_sample:
        # For single-sample overfitting, don't use SceneAwareSampler (Subset doesn't have scenename)
        train_sampler = None  # Will use default sequential sampling
        log_string('Single-sample mode: using default sampler')
    else:
        # Use scene-aware sampling for cache-friendly data loading
        train_sampler = SceneAwareSampler(train_dataset, shuffle=True)
        log_string('Using SceneAwareSampler for cache-friendly collision label loading')
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfgs.batch_size, 
                                  shuffle=False,  # Sampler handles shuffling
                                  sampler=train_sampler,
                                  num_workers=cfgs.num_workers, pin_memory=True, 
                                  persistent_workers=(cfgs.persistent_workers and cfgs.num_workers > 0),
                                  worker_init_fn=my_worker_init_fn, collate_fn=spconv_collate_fn)
    log_string('train dataloader length: ' + str(len(train_dataloader)))
    
    return train_dataloader, train_sampler


def create_model_and_optimizer():
    """Create model, optimizer, and scaler. Each process creates its own."""
    net = GraspNet(
        seed_feat_dim=cfgs.seed_feat_dim, 
        is_training=True, 
        backbone=cfgs.backbone,
        ptv3_pretrained_path=cfgs.ptv3_pretrained_path,
        enable_flash=cfgs.enable_flash,
        enable_stable_score=cfgs.enable_stable_score,
    )
    
    # Set device based on distributed or single-GPU mode
    if is_distributed():
        device = torch.device(f"cuda:{LOCAL_RANK}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net.to(device)
    
    if cfgs.enable_stable_score:
        log_string(f"Stable score prediction enabled (lambda_stable={cfgs.lambda_stable})")

    backbone_lr = cfgs.learning_rate * cfgs.backbone_lr_scale
    head_lr = cfgs.learning_rate
    
    # Separate backbone params from head params for discriminative learning rates
    backbone_decay_params = []
    backbone_no_decay_params = []
    head_decay_params = []
    head_no_decay_params = []
    
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if this is a backbone parameter (excluding output_proj which is randomly initialized)
        # output_proj is our added projection layer to match output dims - it's not pretrained
        is_backbone = name.startswith('backbone.') and 'output_proj' not in name
        
        # Check if this is a normalization parameter or bias (no weight decay)
        is_norm = any(n in name.lower() for n in ['layernorm', 'layer_norm', 'batchnorm', 'batch_norm', '.bn.', '.norm.', '.norm1.', '.norm2.'])
        is_bias = name.endswith('.bias')
        no_decay = is_norm or is_bias
        
        if is_backbone:
            if no_decay:
                backbone_no_decay_params.append(param)
            else:
                backbone_decay_params.append(param)
        else:
            if no_decay:
                head_no_decay_params.append(param)
            else:
                head_decay_params.append(param)
    
    # Choose optimizer based on weight decay setting
    if cfgs.weight_decay > 0:
        # Use AdamW with parameter groups for proper weight decay handling
        weight_decay = cfgs.weight_decay
        param_groups = [
            {'params': backbone_decay_params, 'lr': backbone_lr, 'weight_decay': weight_decay, 'name': 'backbone_decay'},
            {'params': backbone_no_decay_params, 'lr': backbone_lr, 'weight_decay': 0.0, 'name': 'backbone_no_decay'},
            {'params': head_decay_params, 'lr': head_lr, 'weight_decay': 0.0, 'name': 'head_decay'},
            {'params': head_no_decay_params, 'lr': head_lr, 'weight_decay': 0.0, 'name': 'head_no_decay'},
        ]
        # Remove empty groups
        param_groups = [g for g in param_groups if len(g['params']) > 0]
        optimizer = optim.AdamW(param_groups)
        log_string(f"Optimizer: AdamW with weight_decay={weight_decay} (backbone only, heads=0)")
        log_string(f"Learning rates: backbone={backbone_lr:.6f} (scale={cfgs.backbone_lr_scale}), heads={head_lr:.6f}")
        log_string(f"Param groups: backbone_decay={len(backbone_decay_params)}, backbone_no_decay={len(backbone_no_decay_params)}, "
                   f"head_decay={len(head_decay_params)}, head_no_decay={len(head_no_decay_params)}")
    else:
        # Use simple Adam without weight decay
        optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate)
        log_string(f"Optimizer: Adam with lr={cfgs.learning_rate}")

    # Initialize GradScaler for AMP (to prevent small gradients from underflowing to zero)
    scaler = GradScaler(enabled=cfgs.use_amp and device.type == 'cuda')
    if cfgs.use_amp:
        log_string("Using Automatic Mixed Precision (AMP) training")

    start_epoch = 0
    if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        # Handle both DDP and non-DDP checkpoints
        state_dict = checkpoint['model_state_dict']
        # Remove 'module.' prefix if loading from DDP checkpoint into non-DDP model
        if not is_distributed() and any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # Add 'module.' prefix if loading non-DDP checkpoint into DDP model (handled after DDP wrap)
        net.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and cfgs.use_amp:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))
    
    # Wrap model in DDP if running distributed
    if is_distributed():
        # find_unused_parameters=True is needed because some parameters might not be used 
        # in every forward pass (e.g., stable_score_head when disabled)
        net = DDP(net, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, 
                  find_unused_parameters=True)
        log_string(f"Model wrapped in DistributedDataParallel (device_ids=[{LOCAL_RANK}])")
    
    return net, optimizer, scaler, start_epoch, device


def get_current_lr(epoch, base_lr):
    """Calculate LR with decay (0.95^epoch)."""
    lr = base_lr * (0.95 ** epoch)
    return lr


def adjust_learning_rate(optimizer, epoch):
    """Adjust LR for all param groups, maintaining their relative ratios."""
    head_lr = get_current_lr(epoch, cfgs.learning_rate)
    backbone_lr = get_current_lr(epoch, cfgs.learning_rate * cfgs.backbone_lr_scale)
    
    for param_group in optimizer.param_groups:
        group_name = param_group.get('name', '')
        if 'backbone' in group_name:
            param_group['lr'] = backbone_lr
        else:
            param_group['lr'] = head_lr


def train_one_epoch(net, optimizer, scaler, device, train_dataloader, train_writer):
    stat_dict = {}  # collect statistics
    epoch_stat_dict = {}  # collect epoch-level statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()
    batch_interval = 20
    
    # Only show progress bar on main process to avoid duplicates
    data_iter = tqdm(enumerate(train_dataloader), desc='Training', 
                     disable=not is_main_process(), total=len(train_dataloader))
    
    for batch_idx, batch_data_label in data_iter:
            
        # Transfer to GPU with non_blocking=True for async copy (works with pin_memory=True)
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device, non_blocking=True)
            else:
                batch_data_label[key] = batch_data_label[key].to(device, non_blocking=True)

        # Forward pass with autocast for mixed precision
        with autocast(enabled=cfgs.use_amp, device_type=device.type):
            end_points = net(batch_data_label)
            loss, end_points = get_loss(end_points, 
                                        enable_stable_score=cfgs.enable_stable_score,
                                        lambda_stable=cfgs.lambda_stable)
            # Scale loss for gradient accumulation
            loss = loss / cfgs.accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Only step optimizer every accumulation_steps
        if (batch_idx + 1) % cfgs.accumulation_steps == 0:
            # Gradient clipping (important for transformer stability)
            if cfgs.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfgs.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        """# Periodic cache clearing to prevent memory fragmentation
        if (batch_idx + 1) % 100 == 0:
            torch.cuda.empty_cache()"""

        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                if key not in epoch_stat_dict:
                    epoch_stat_dict[key] = 0
                # Multiply back by accumulation_steps to get actual loss value
                loss_value = end_points[key].item() if 'loss' not in key else end_points[key].item() * cfgs.accumulation_steps
                stat_dict[key] += loss_value
                epoch_stat_dict[key] += loss_value

        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ----epoch: %03d  ---- batch: %03d ----' % (EPOCH_CNT, batch_idx + 1))
            for key in sorted(stat_dict.keys()):
                if is_main_process() and train_writer is not None:
                    train_writer.add_scalar(key, stat_dict[key] / batch_interval,
                                            (EPOCH_CNT * len(train_dataloader) + batch_idx) * cfgs.batch_size)
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0
    
    # Log epoch-level averages to TensorBoard (only main process)
    num_batches = len(train_dataloader)
    if is_main_process() and train_writer is not None:
        for key in sorted(epoch_stat_dict.keys()):
            avg_value = epoch_stat_dict[key] / num_batches
            train_writer.add_scalar('epoch_' + key, avg_value, EPOCH_CNT)
        
        # Flush to ensure data is written to disk
        train_writer.flush()
    
    # Return epoch average loss
    return epoch_stat_dict['loss/overall_loss'] / num_batches if 'loss/overall_loss' in epoch_stat_dict else 0


def train(start_epoch, net, optimizer, scaler, device, train_dataloader,
          train_writer, train_sampler=None):
    global EPOCH_CNT
    
    for epoch in range(start_epoch, cfgs.max_epoch):
        # Set epoch on distributed sampler for proper shuffling
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % epoch)
        log_string('Current learning rate: head=%.6f, backbone=%.6f' % (
            get_current_lr(epoch, cfgs.learning_rate),
            get_current_lr(epoch, cfgs.learning_rate * cfgs.backbone_lr_scale)
        ))
        log_string(str(datetime.now()))
        
        # Log learning rate to TensorBoard (only main process)
        if is_main_process():
            train_writer.add_scalar('learning_rate/head', get_current_lr(epoch, cfgs.learning_rate), epoch)
            train_writer.add_scalar('learning_rate/backbone', get_current_lr(epoch, cfgs.learning_rate * cfgs.backbone_lr_scale), epoch)
        
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_loss = train_one_epoch(net, optimizer, scaler, device, train_dataloader, train_writer)
        
        # Save regular checkpoint (only from main process)
        if is_main_process():
            # Get the underlying model if wrapped in DDP
            model_to_save = net.module if is_distributed() else net
            save_dict = {'epoch': epoch + 1, 'optimizer_state_dict': optimizer.state_dict(),
                         'model_state_dict': model_to_save.state_dict(),
                         'scaler_state_dict': scaler.state_dict()}
            torch.save(save_dict, os.path.join(cfgs.log_dir, cfgs.model_name + '_epoch' + str(epoch + 1).zfill(2) + '.tar'))


if __name__ == '__main__':
    # Create dataloaders (each process creates its own for DDP)
    TRAIN_DATALOADER, TRAIN_SAMPLER= create_dataloaders()
    
    # Create model, optimizer, and scaler (each process creates its own for DDP)
    net, optimizer, scaler, start_epoch, device = create_model_and_optimizer()
    
    # TensorBoard Visualizers (only main process writes to TensorBoard)
    if is_main_process():
        TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))
        
    else:
        # Create dummy writers for non-main processes (won't be used but simplifies code)
        TRAIN_WRITER = None
        
    
    # Start training
    try:
        train(start_epoch, net, optimizer, scaler, device, TRAIN_DATALOADER,
              TRAIN_WRITER, TRAIN_SAMPLER)
    finally:
        # Ensure TensorBoard writers are properly closed
        if is_main_process():
            TRAIN_WRITER.close()
           
        # Clean up distributed training
        cleanup_distributed()
