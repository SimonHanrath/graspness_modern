import os
import sys

# Fix spconv algorithm tuner crash on Ada GPUs (L40S, RTX 4090, etc.)
# Must be set BEFORE any spconv import
os.environ.setdefault('SPCONV_ALGO', 'Native')  # Use native algorithm, not implicit gemm
os.environ.setdefault('CUMM_CUDA_ARCH_LIST', '8.9')  # Ada Lovelace compute capability
os.environ.setdefault('SPCONV_DISABLE_JIT', '1')  # Disable JIT to avoid profiling crashes

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


def freeze_for_stable_finetune(net, log_fn=print):
    """Freeze all parameters except the stable score head (conv_stable).
    
    Used when fine-tuning a pretrained model to add stable score prediction.
    """
    frozen_count = 0
    trainable_count = 0
    
    for name, param in net.named_parameters():
        if 'conv_stable' in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    
    log_fn(f"Frozen {frozen_count:,} params, training {trainable_count:,} params (conv_stable only)")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--train_split', default='train', help='Training split [train/train_all]')
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
parser.add_argument('--backbone', type=str, default='transformer', choices=['transformer', 'transformer_pretrained', 'sonata', 'pointnet2', 'resunet', 'resunet18', 'resunet_rgb', 'resunet18_rgb'],
                    help='Backbone architecture [default: transformer]. resunet=14D, resunet18=18D (more layers). sonata=self-supervised PTv3 (CVPR 2025). Use _rgb suffix for 6-channel RGB input.')
parser.add_argument('--grad_clip', type=float, default=0.0,
                    help='Gradient clipping max norm (recommended: 1.0-5.0 for transformers, 0 to disable) [default: 0.0]')
parser.add_argument('--ptv3_pretrained_path', type=str, default=None,
                    help='Path to PTv3 pretrained weights (.pth file). If not specified, uses models/pointcept/model_best.pth')
parser.add_argument('--enable_flash', action='store_true', default=False,
                    help='Enable flash attention in PTv3 backbone (requires flash_attn package)')
parser.add_argument('--accumulation_steps', type=int, default=1,
                    help='Gradient accumulation steps (simulate larger batch with batch_size=1) [default: 1]')
parser.add_argument('--backbone_lr_scale', type=float, default=None,
                    help='Learning rate multiplier for backbone (e.g., 0.1 for pretrained). Default: 0.1 for transformer_pretrained/sonata, 1.0 otherwise')
parser.add_argument('--layer_decay', type=float, default=None,
                    help='Layer-wise LR decay factor for pretrained backbones. Each encoder stage gets lr * layer_decay^(num_stages - stage). '
                         'Default: 0.65 for sonata/transformer_pretrained, 1.0 (disabled) otherwise')
parser.add_argument('--enable_stable_score', action='store_true', default=False,
                    help='Enable stable score prediction to penalize grasps that may cause tipping [default: False]')
parser.add_argument('--view_start', type=int, default=0,
                    help='Starting view index (inclusive) for each scene [default: 0]')
parser.add_argument('--view_end', type=int, default=256,
                    help='Ending view index (exclusive) for each scene [default: 256]')
parser.add_argument('--include_floor', action='store_true', default=False,
                    help='Include floor/table points in training (uses graspness_full/ labels, requires running generate_graspness_full.py first)')
parser.add_argument('--lambda_stable', type=float, default=10.0,
                    help='Weight for stable score loss term [default: 10.0]')
parser.add_argument('--graspness_threshold', type=float, default=0.1,
                    help='Threshold for graspness score filtering during forward pass [default: -0.1]')
parser.add_argument('--nsample', type=int, default=16,
                    help='Number of samples for cloud crop in GraspNet [default: 16]')
parser.add_argument('--cosine_lr', action='store_true', default=False,
                    help='Use cosine annealing LR schedule with warmup instead of exponential decay')
parser.add_argument('--warmup_epochs', type=int, default=2,
                    help='Number of warmup epochs for cosine LR schedule [default: 2]')
parser.add_argument('--finetune', action='store_true', default=False,
                    help='Fine-tune mode: load weights but reset epoch to 0 and skip optimizer state. Use with --checkpoint_path to fine-tune a vanilla model with stable score.')
parser.add_argument('--debug_feature_stats', action='store_true', default=False,
                    help='Print mean/std feature statistics at each backbone stage for first forward pass [default: False]')
parser.add_argument('--no_translation_aug', action='store_true', default=False,
                    help='Disable random translation in data augmentation (paper uses no translation) [default: False]')
parser.add_argument('--use_val', action='store_true', default=False,
                    help='Enable validation: use train_reduced (95 scenes) for training and val_train (5 scenes) for validation [default: False]')
# DDP arguments (set automatically by torchrun, but can be overridden)
parser.add_argument('--local_rank', type=int, default=-1,
                    help='Local rank for distributed training (set by torchrun)')


cfgs = parser.parse_args()

# Set layer_decay default (0.65 for pretrained transformers, 1.0 = disabled otherwise)
if cfgs.layer_decay is None:
    cfgs.layer_decay = 0.65 if cfgs.backbone in ('transformer_pretrained', 'sonata') else 1.0

# Set backbone_lr_scale default: 1.0 when LLRD is active (it handles the scaling),
# 0.1 for pretrained without LLRD, 1.0 for non-pretrained
if cfgs.backbone_lr_scale is None:
    if cfgs.layer_decay < 1.0:
        cfgs.backbone_lr_scale = 1.0  # LLRD handles per-stage scaling
    elif cfgs.backbone in ('transformer_pretrained', 'sonata'):
        cfgs.backbone_lr_scale = 0.1  # flat scaling fallback
    else:
        cfgs.backbone_lr_scale = 1.0

# Auto-enable cosine LR with warmup for pretrained backbones (unless user explicitly set cosine_lr)
_is_pretrained_backbone = cfgs.backbone in ('transformer_pretrained', 'sonata')
if _is_pretrained_backbone and not cfgs.cosine_lr:
    cfgs.cosine_lr = True

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
# Load checkpoint if resuming OR fine-tuning
CHECKPOINT_PATH = cfgs.checkpoint_path if (cfgs.resume or cfgs.finetune) else None
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

    use_rgb = (cfgs.backbone in ['transformer_pretrained', 'resunet_rgb'])
    
    # Determine training split (use train_reduced if validation is enabled)
    actual_train_split = 'train_reduced' if cfgs.use_val and cfgs.train_split == 'train' else cfgs.train_split
    
    train_dataset = GraspNetDataset(cfgs.dataset_root, grasp_labels=grasp_labels, camera=cfgs.camera, split=actual_train_split,
                                    num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                    remove_outlier=True, augment=True, load_label=True, use_rgb=use_rgb,
                                    enable_stable_score=cfgs.enable_stable_score, view_start=cfgs.view_start, view_end=cfgs.view_end,
                                    include_floor=cfgs.include_floor, augment_translation=not cfgs.no_translation_aug)
    log_string(f'train dataset length: {len(train_dataset)} (split: {actual_train_split})')


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
    
    # Create validation dataloader if enabled
    val_dataloader = None
    if cfgs.use_val:
        val_dataset = GraspNetDataset(cfgs.dataset_root, grasp_labels=grasp_labels, camera=cfgs.camera, split='val_train',
                                      num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                      remove_outlier=True, augment=False, load_label=True, use_rgb=use_rgb,
                                      enable_stable_score=cfgs.enable_stable_score, view_start=cfgs.view_start, view_end=cfgs.view_end,
                                      include_floor=cfgs.include_floor, augment_translation=False)
        log_string(f'val dataset length: {len(val_dataset)} (split: val_train, scenes 95-99)')
        
        val_dataloader = DataLoader(val_dataset, batch_size=cfgs.batch_size,
                                    shuffle=False, num_workers=cfgs.num_workers, pin_memory=True,
                                    persistent_workers=(cfgs.persistent_workers and cfgs.num_workers > 0),
                                    worker_init_fn=my_worker_init_fn, collate_fn=spconv_collate_fn)
        log_string('val dataloader length: ' + str(len(val_dataloader)))
    
    return train_dataloader, train_sampler, val_dataloader


def create_model_and_optimizer():
    """Create model, optimizer, and scaler. Each process creates its own."""
    net = GraspNet(
        seed_feat_dim=cfgs.seed_feat_dim, 
        is_training=True, 
        backbone=cfgs.backbone,
        ptv3_pretrained_path=cfgs.ptv3_pretrained_path,
        enable_flash=cfgs.enable_flash,
        enable_stable_score=cfgs.enable_stable_score,
        graspness_threshold=cfgs.graspness_threshold,
        nsample=cfgs.nsample,
        debug_feature_stats=cfgs.debug_feature_stats,
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

    head_lr = cfgs.learning_rate
    backbone_base_lr = cfgs.learning_rate * cfgs.backbone_lr_scale
    layer_decay = cfgs.layer_decay
    weight_decay = cfgs.weight_decay if cfgs.weight_decay > 0 else 0.0
    
    def _get_backbone_stage(name):
        """Return the encoder stage index for a backbone parameter, or -1 for embedding/fusion.
        
        Sonata:  backbone.encoder.enc.enc{0-4}.block*  / backbone.encoder.embedding.*
        PTv3:    backbone.enc.enc{0-4}.block*  / backbone.dec.dec{0-3}.* / backbone.embedding.*
        """
        import re
        # Encoder stages (both Sonata and PTv3)
        m = re.search(r'\.enc\.enc(\d+)\.', name)
        if m:
            return int(m.group(1))
        # PTv3 decoder stages — treat at same depth as the encoder stage they mirror
        # dec3 mirrors enc4, dec2 mirrors enc3, dec1 mirrors enc2, dec0 mirrors enc1
        m = re.search(r'\.dec\.dec(\d+)\.', name)
        if m:
            return int(m.group(1)) + 1  # dec3→stage4, dec2→stage3, etc.
        # Everything else (embedding, fusion_proj) → stage -1 (gets lowest LR)
        return -1
    
    # Determine number of encoder stages from the backbone
    num_enc_stages = 5  # Both Sonata and PTv3 have enc0..enc4
    
    # Build per-stage parameter groups for LLRD
    # Stage LR: backbone_base_lr * layer_decay^(num_enc_stages - stage)
    # Embedding (stage -1): backbone_base_lr * layer_decay^(num_enc_stages + 1)  (lowest)
    # fusion_proj: treated as head (randomly initialized, like output_proj)
    head_decay_params = []
    head_no_decay_params = []
    # Dict: stage_idx -> {'decay': [...], 'no_decay': [...]}
    backbone_stage_params = {}
    
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        
        # output_proj and fusion_proj are randomly initialized projection layers → head LR
        is_backbone = name.startswith('backbone.') and 'output_proj' not in name and 'fusion_proj' not in name
        
        is_norm = any(n in name.lower() for n in ['layernorm', 'layer_norm', 'batchnorm', 'batch_norm', '.bn.', '.norm.', '.norm1.', '.norm2.'])
        is_bias = name.endswith('.bias')
        no_decay = is_norm or is_bias
        
        if is_backbone:
            stage = _get_backbone_stage(name)
            if stage not in backbone_stage_params:
                backbone_stage_params[stage] = {'decay': [], 'no_decay': []}
            if no_decay:
                backbone_stage_params[stage]['no_decay'].append(param)
            else:
                backbone_stage_params[stage]['decay'].append(param)
        else:
            if no_decay:
                head_no_decay_params.append(param)
            else:
                head_decay_params.append(param)
    
    # Build optimizer param groups
    param_groups = []
    
    # Backbone groups with per-stage LR (LLRD)
    for stage in sorted(backbone_stage_params.keys()):
        if stage == -1:
            # Embedding — deepest decay
            stage_scale = layer_decay ** (num_enc_stages + 1)
            stage_name = 'embedding'
        else:
            # Encoder/decoder stage — deeper stages get higher LR
            stage_scale = layer_decay ** (num_enc_stages - stage)
            stage_name = f'stage{stage}'
        
        stage_lr = backbone_base_lr * stage_scale
        
        if backbone_stage_params[stage]['decay']:
            param_groups.append({
                'params': backbone_stage_params[stage]['decay'],
                'lr': stage_lr,
                'weight_decay': weight_decay,
                'name': f'backbone_{stage_name}_decay',
                'backbone_stage': stage,
            })
        if backbone_stage_params[stage]['no_decay']:
            param_groups.append({
                'params': backbone_stage_params[stage]['no_decay'],
                'lr': stage_lr,
                'weight_decay': 0.0,
                'name': f'backbone_{stage_name}_no_decay',
                'backbone_stage': stage,
            })
    
    # Head groups
    if head_decay_params:
        param_groups.append({'params': head_decay_params, 'lr': head_lr, 'weight_decay': 0.0, 'name': 'head_decay'})
    if head_no_decay_params:
        param_groups.append({'params': head_no_decay_params, 'lr': head_lr, 'weight_decay': 0.0, 'name': 'head_no_decay'})
    
    if cfgs.weight_decay > 0:
        optimizer = optim.AdamW(param_groups)
        log_string(f"Optimizer: AdamW with weight_decay={weight_decay}")
    else:
        optimizer = optim.Adam(param_groups)
        log_string(f"Optimizer: Adam (no weight decay)")
    
    log_string(f"Learning rates: backbone_base={backbone_base_lr:.6f} (scale={cfgs.backbone_lr_scale}), "
               f"heads={head_lr:.6f}, layer_decay={layer_decay}")
    log_string(f"LR schedule: {'cosine + warmup(' + str(cfgs.warmup_epochs) + ' epochs)' if cfgs.cosine_lr else 'exponential (0.95^epoch)'}")
    for pg in param_groups:
        log_string(f"  {pg['name']}: {len(pg['params'])} params, lr={pg['lr']:.8f}")

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
        
        net.load_state_dict(state_dict, strict=False)
        
        # In finetune mode, skip optimizer state and reset epoch
        if cfgs.finetune:
            log_string("-> FINETUNE mode: loaded weights from %s, resetting epoch to 0" % CHECKPOINT_PATH)
            # Freeze all except stable score head when fine-tuning for stable score
            if cfgs.enable_stable_score:
                freeze_for_stable_finetune(net, log_string)
        else:
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
    """Calculate LR based on schedule (exponential or cosine with warmup)."""
    if cfgs.cosine_lr:
        # Cosine annealing with linear warmup
        if epoch < cfgs.warmup_epochs:
            return base_lr * (epoch + 1) / cfgs.warmup_epochs
        else:
            import math
            progress = (epoch - cfgs.warmup_epochs) / max(1, cfgs.max_epoch - cfgs.warmup_epochs)
            return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        # Exponential decay (original)
        return base_lr * (0.95 ** epoch)


def adjust_learning_rate(optimizer, epoch):
    """Adjust LR for all param groups, maintaining their relative LLRD ratios."""
    head_lr = get_current_lr(epoch, cfgs.learning_rate)
    backbone_base_lr = get_current_lr(epoch, cfgs.learning_rate * cfgs.backbone_lr_scale)
    num_enc_stages = 5
    
    for param_group in optimizer.param_groups:
        group_name = param_group.get('name', '')
        if 'backbone' in group_name:
            stage = param_group.get('backbone_stage', 0)
            if stage == -1:
                stage_scale = cfgs.layer_decay ** (num_enc_stages + 1)
            else:
                stage_scale = cfgs.layer_decay ** (num_enc_stages - stage)
            param_group['lr'] = backbone_base_lr * stage_scale
        else:
            param_group['lr'] = head_lr


def train_one_epoch(net, optimizer, scaler, device, train_dataloader, train_writer):
    stat_dict = {}  # collect statistics
    epoch_stat_dict = {}  # collect epoch-level statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()
    batch_interval = 20
    
    # Zero gradients at the start of each epoch to ensure clean state
    optimizer.zero_grad()
    
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

        # Skip batches with too few voxels (would cause spconv errors after 16x downsampling)
        # ResUNet has 4 stride-2 convs, so we need enough voxels to survive downsampling
        MIN_VOXELS = 64
        if 'coors' in batch_data_label:
            num_voxels = batch_data_label['coors'].shape[0]
            if num_voxels < MIN_VOXELS:
                log_string(f'[Train] Skipping batch {batch_idx}: too few voxels ({num_voxels} < {MIN_VOXELS})')
                continue

        # Forward pass with autocast for mixed precision
        try:
            with autocast(enabled=cfgs.use_amp, device_type=device.type):
                end_points = net(batch_data_label)
                loss, end_points = get_loss(end_points, 
                                            enable_stable_score=cfgs.enable_stable_score,
                                            lambda_stable=cfgs.lambda_stable)
                # Scale loss for gradient accumulation
                loss = loss / cfgs.accumulation_steps
        except RuntimeError as e:
            if "can't find suitable algorithm" in str(e) or "assert faild" in str(e):
                log_string(f'[Train] Skipping batch {batch_idx}: spconv error - {e}')
                continue
            raise  # Re-raise other RuntimeErrors
        
        # Debug: Print backbone and graspness statistics for first batch of first epoch
        if batch_idx == 0 and EPOCH_CNT == 0 and is_main_process():
            print("\n" + "="*80)
            print("DEBUG: Backbone & Graspness Statistics (Epoch 0, Batch 0)")
            print("="*80)
            print(f"Backbone features:")
            print(f"  mean={end_points['backbone_feat_mean'].item():.6f}")
            print(f"  std={end_points['backbone_feat_std'].item():.6f}")
            print(f"  min={end_points['backbone_feat_min'].item():.6f}")
            print(f"  max={end_points['backbone_feat_max'].item():.6f}")
            print(f"  abs_mean={end_points['backbone_feat_abs_mean'].item():.6f}")
            graspness = end_points['graspness_score']
            print(f"\nGraspness scores:")
            print(f"  mean={graspness.mean().item():.6f}")
            print(f"  std={graspness.std().item():.6f}")
            print(f"  min={graspness.min().item():.6f}")
            print(f"  max={graspness.max().item():.6f}")
            print(f"  >0.01: {(graspness > 0.01).sum().item()} / {graspness.numel()} ({100*(graspness > 0.01).float().mean().item():.2f}%)")
            print(f"  >0.1:  {(graspness > 0.1).sum().item()} / {graspness.numel()} ({100*(graspness > 0.1).float().mean().item():.2f}%)")
            print(f"  >0.3:  {(graspness > 0.3).sum().item()} / {graspness.numel()} ({100*(graspness > 0.3).float().mean().item():.2f}%)")
            print(f"\nGraspable count (after threshold): {end_points['graspable_count_stage1'].item():.1f}")
            print("="*80 + "\n")
            
            # Save raw distribution data for later plotting
            try:
                graspness_np = graspness.detach().cpu().numpy().flatten()
                data_path = os.path.join('tmp', 'init_distribution_data.npz')
                np.savez(data_path,
                    graspness_scores=graspness_np,
                    backbone_mean=end_points['backbone_feat_mean'].item(),
                    backbone_std=end_points['backbone_feat_std'].item(),
                    backbone_min=end_points['backbone_feat_min'].item(),
                    backbone_max=end_points['backbone_feat_max'].item(),
                    backbone_abs_mean=end_points['backbone_feat_abs_mean'].item(),
                )
                print(f"Saved distribution data to: {data_path}")
                print("To plot, run: python experiments/plots/plot_init_distribution.py --data <path>")
            except Exception as e:
                print(f"Warning: Could not save distribution data: {e}")
        
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
                loss_value = end_points[key].item()
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
    
    # Handle remaining gradients if num_batches not divisible by accumulation_steps
    if len(train_dataloader) % cfgs.accumulation_steps != 0:
        if cfgs.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfgs.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
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


@torch.no_grad()
def validate_one_epoch(net, device, val_dataloader, val_writer):
    """Run validation and return average loss."""
    stat_dict = {}
    successful_batches = 0
    skipped_batches = 0
    
    # Try to clear spconv's algorithm cache to avoid stale tuning results
    try:
        import spconv.pytorch as spconv
        if hasattr(spconv, 'algocore'):
            spconv.algocore.clear_cache()
            log_string('[Val] Cleared spconv algorithm cache')
    except Exception as e:
        pass  # Older spconv versions may not have this
    
    net.eval()
    
    data_iter = tqdm(enumerate(val_dataloader), desc='Validation',
                     disable=not is_main_process(), total=len(val_dataloader))
    
    for batch_idx, batch_data_label in data_iter:
        # Transfer to GPU
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device, non_blocking=True)
            else:
                batch_data_label[key] = batch_data_label[key].to(device, non_blocking=True)
        
        # Skip batches with too few voxels (would cause spconv errors after 16x downsampling)
        # ResUNet has 4 stride-2 convs, so we need enough voxels to survive downsampling
        MIN_VOXELS = 64
        if 'coors' in batch_data_label:
            num_voxels = batch_data_label['coors'].shape[0]
            if num_voxels < MIN_VOXELS:
                log_string(f'[Val] Skipping batch {batch_idx}: too few voxels ({num_voxels} < {MIN_VOXELS})')
                skipped_batches += 1
                continue
        
        # Debug: Log voxel count for first few batches and any that fail
        if batch_idx < 5:
            log_string(f'[Val Debug] Batch {batch_idx}: {batch_data_label["coors"].shape[0]} voxels')
        
        # Forward pass with error handling for spconv edge cases
        try:
            with autocast(enabled=cfgs.use_amp, device_type=device.type):
                end_points = net(batch_data_label)
                loss, end_points = get_loss(end_points,
                                            enable_stable_score=cfgs.enable_stable_score,
                                            lambda_stable=cfgs.lambda_stable)
        except RuntimeError as e:
            if "can't find suitable algorithm" in str(e) or "assert faild" in str(e):
                log_string(f'[Val] Skipping batch {batch_idx}: spconv error ({batch_data_label["coors"].shape[0]} voxels) - {e}')
                skipped_batches += 1
                continue
            raise  # Re-raise other RuntimeErrors
        
        successful_batches += 1
        # Accumulate statistics
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()
    
    # Compute averages and log
    total_batches = len(val_dataloader)
    log_string(' ---- Validation Results ----')
    log_string(f'Processed {successful_batches}/{total_batches} batches (skipped {skipped_batches})')
    
    if successful_batches == 0:
        log_string('WARNING: All validation batches failed! Check data/model compatibility.')
        return float('inf')
    
    for key in sorted(stat_dict.keys()):
        avg_value = stat_dict[key] / successful_batches  # Divide by successful, not total
        log_string('val %s: %f' % (key, avg_value))
        if is_main_process() and val_writer is not None:
           val_writer.add_scalar('epoch_' + key, avg_value, EPOCH_CNT)
    
    if is_main_process() and val_writer is not None:
        val_writer.flush()
    
    return stat_dict.get('loss/overall_loss', 0) / successful_batches


def train(start_epoch, net, optimizer, scaler, device, train_dataloader,
          train_writer, train_sampler=None, val_dataloader=None, val_writer=None):
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
        
        # Run validation if enabled
        if val_dataloader is not None:
            val_loss = validate_one_epoch(net, device, val_dataloader, val_writer)
            log_string(f'Epoch {epoch} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
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
    TRAIN_DATALOADER, TRAIN_SAMPLER, VAL_DATALOADER = create_dataloaders()
    
    # Create model, optimizer, and scaler (each process creates its own for DDP)
    net, optimizer, scaler, start_epoch, device = create_model_and_optimizer()
    
    # DIAGNOSTIC: Quick test of validation forward pass before full training
    if VAL_DATALOADER is not None and is_main_process():
        log_string("=== DIAGNOSTIC: Testing validation forward pass ===")
        net.eval()
        with torch.no_grad():
            for i, batch in enumerate(VAL_DATALOADER):
                if i >= 3:  # Test first 3 batches only
                    break
                for key in batch:
                    if 'list' not in key:
                        batch[key] = batch[key].to(device)
                    else:
                        for j in range(len(batch[key])):
                            for k in range(len(batch[key][j])):
                                batch[key][j][k] = batch[key][j][k].to(device)
                log_string(f"  Val batch {i}: {batch['coors'].shape[0]} voxels")
                try:
                    with autocast(enabled=cfgs.use_amp, device_type=device.type):
                        end_points = net(batch)
                    log_string(f"  Val batch {i}: SUCCESS")
                except RuntimeError as e:
                    log_string(f"  Val batch {i}: FAILED - {e}")
                    break  # Stop on first failure
        net.train()
        log_string("=== END DIAGNOSTIC ===")
    
    # TensorBoard Visualizers (only main process writes to TensorBoard)
    if is_main_process():
        TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))
        VAL_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'val')) if cfgs.use_val else None
    else:
        # Create dummy writers for non-main processes (won't be used but simplifies code)
        TRAIN_WRITER = None
        VAL_WRITER = None
    
    # Start training
    try:
        train(start_epoch, net, optimizer, scaler, device, TRAIN_DATALOADER,
              TRAIN_WRITER, TRAIN_SAMPLER, VAL_DATALOADER, VAL_WRITER)
    finally:
        # Ensure TensorBoard writers are properly closed
        if is_main_process():
            TRAIN_WRITER.close()
            if VAL_WRITER is not None:
                VAL_WRITER.close()
        
        # Clean up distributed training
        cleanup_distributed()
