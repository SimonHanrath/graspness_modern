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
from dataset.graspnet_dataset import GraspNetDataset, spconv_collate_fn, load_grasp_labels, load_grasp_labels_lazy

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
parser.add_argument('--val_split', type=str, default='val', choices=['val', 'test_seen'], 
                    help='Validation split: "val" uses scenes 7-8 (has labels), "test_seen" uses scene 10 (needs label generation) [default: val]')
parser.add_argument('--use_compile', action='store_true', default=False, help='Use torch.compile for model optimization [PyTorch 2.0+]')
parser.add_argument('--use_amp', action='store_true', default=False,
                    help='Use torch.cuda.amp for mixed-precision training')
parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers [default: 0]')
parser.add_argument('--persistent_workers', action='store_true', default=False, 
                    help='Keep workers alive between epochs (reduces memory overhead with num_workers>0)')
parser.add_argument('--lazy_grasp_labels', action='store_true', default=False,
                    help='Use lazy loading for grasp labels to reduce memory (useful with many workers)')


cfgs = parser.parse_args()


EPOCH_CNT = 0
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None and cfgs.resume else None
if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def create_dataloaders():
    """Create datasets and dataloaders. Only called in main process."""
    # Load grasp labels (use lazy loading if specified to save memory with multiple workers)
    if cfgs.lazy_grasp_labels:
        log_string("Using lazy loading for grasp labels (memory-efficient mode)")
        grasp_labels = load_grasp_labels_lazy(cfgs.dataset_root)
    else:
        log_string("Loading all grasp labels into memory (~21GB)")
        grasp_labels = load_grasp_labels(cfgs.dataset_root)

    train_dataset = GraspNetDataset(cfgs.dataset_root, grasp_labels=grasp_labels, camera=cfgs.camera, split='train',
                                    num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                    remove_outlier=True, augment=True, load_label=True)
    log_string('train dataset length: ' + str(len(train_dataset)))

    # Validation dataset (use specified validation split without augmentation)
    val_dataset = GraspNetDataset(cfgs.dataset_root, grasp_labels=grasp_labels, camera=cfgs.camera, split=cfgs.val_split,
                                  num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                  remove_outlier=True, augment=False, load_label=True)
    log_string('validation dataset length: ' + str(len(val_dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=cfgs.batch_size, shuffle=True,
                                  num_workers=cfgs.num_workers, pin_memory=True, 
                                  persistent_workers=(cfgs.persistent_workers and cfgs.num_workers > 0),
                                  worker_init_fn=my_worker_init_fn, collate_fn=spconv_collate_fn)
    log_string('train dataloader length: ' + str(len(train_dataloader)))

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=cfgs.num_workers, pin_memory=True,
                                persistent_workers=(cfgs.persistent_workers and cfgs.num_workers > 0),
                                worker_init_fn=my_worker_init_fn, collate_fn=spconv_collate_fn)
    log_string('validation dataloader length: ' + str(len(val_dataloader)))
    
    return train_dataloader, val_dataloader


def create_model_and_optimizer():
    """Create model, optimizer, and scaler. Only called in main process."""
    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Apply torch.compile for graph optimization ((requires CUDA capability >= 7.0)
    if cfgs.use_compile:
        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability(device)
            cc_major, cc_minor = compute_capability
            cc_value = cc_major + cc_minor * 0.1
            
            if cc_value >= 7.0:
                log_string(f"Compiling model with torch.compile (GPU compute capability: {cc_major}.{cc_minor})...")
                net = torch.compile(net, mode='default')
                log_string("Model compilation enabled. First iteration will be slower.")
            else:
                gpu_name = torch.cuda.get_device_name(device)
                log_string(f"WARNING: torch.compile disabled - {gpu_name} (compute capability {cc_major}.{cc_minor}) is not supported.")
                log_string(f"         Triton compiler requires CUDA capability >= 7.0. Training will continue without compilation.")
        else:
            log_string("Compiling model with torch.compile (mode='default')...")
            net = torch.compile(net, mode='default')
            log_string("Model compilation enabled. First iteration will be slower.")

    # Load the Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate)

    # Initialize GradScaler for AMP (to prevent small gradients from underflowing to zero)
    scaler = GradScaler(enabled=cfgs.use_amp and device.type == 'cuda')
    if cfgs.use_amp:
        log_string("Using Automatic Mixed Precision (AMP) training")

    start_epoch = 0
    if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and cfgs.use_amp:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))
    
    return net, optimizer, scaler, start_epoch, device


def get_current_lr(epoch):
    lr = cfgs.learning_rate
    lr = lr * (0.95 ** epoch)
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch(net, optimizer, scaler, device, train_dataloader, train_writer):
    stat_dict = {}  # collect statistics
    epoch_stat_dict = {}  # collect epoch-level statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()
    batch_interval = 20
    
    # TEST: Start from batch 8000 to test the spatial shape fix
    #START_BATCH = 8000
    #END_BATCH = 8400
    
    for batch_idx, batch_data_label in tqdm(enumerate(train_dataloader), desc='Training'):
        # Skip batches before START_BATCH
        """if batch_idx < START_BATCH:
            continue
        # Stop after END_BATCH
        if batch_idx >= END_BATCH:
            print(f"\n✓ Successfully tested batches {START_BATCH} to {END_BATCH-1}")
            break"""
            
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass with autocast for mixed precision
        with autocast(enabled=cfgs.use_amp, device_type=device.type):
            end_points = net(batch_data_label)
            loss, end_points = get_loss(end_points)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
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
                stat_dict[key] += end_points[key].item()
                epoch_stat_dict[key] += end_points[key].item()

        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ----epoch: %03d  ---- batch: %03d ----' % (EPOCH_CNT, batch_idx + 1))
            for key in sorted(stat_dict.keys()):
                train_writer.add_scalar(key, stat_dict[key] / batch_interval,
                                        (EPOCH_CNT * len(train_dataloader) + batch_idx) * cfgs.batch_size)
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0
    
    # Log epoch-level averages to TensorBoard
    num_batches = len(train_dataloader)
    for key in sorted(epoch_stat_dict.keys()):
        avg_value = epoch_stat_dict[key] / num_batches
        train_writer.add_scalar('epoch_' + key, avg_value, EPOCH_CNT)
    
    # Return epoch average loss
    return epoch_stat_dict['loss/overall_loss'] / num_batches if 'loss/overall_loss' in epoch_stat_dict else 0


def validate_one_epoch(net, device, val_dataloader, val_writer):
    """Run validation and return average loss and statistics"""
    stat_dict = {}  # collect statistics
    net.eval()
    
    with torch.inference_mode():
        for batch_idx, batch_data_label in tqdm(enumerate(val_dataloader), desc='Validating'):
            for key in batch_data_label:
                if 'list' in key:
                    for i in range(len(batch_data_label[key])):
                        for j in range(len(batch_data_label[key][i])):
                            batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
                else:
                    batch_data_label[key] = batch_data_label[key].to(device)

            # Use autocast for validation as well
            with autocast(enabled=cfgs.use_amp):
                end_points = net(batch_data_label)
                loss, end_points = get_loss(end_points)

            for key in end_points:
                if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                    if key not in stat_dict:
                        stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()
    
    # Calculate averages and log to TensorBoard
    num_batches = len(val_dataloader)
    log_string('---- Validation Results ----')
    for key in sorted(stat_dict.keys()):
        avg_value = stat_dict[key] / num_batches
        # Log with 'epoch_' prefix to match training metrics for comparison
        val_writer.add_scalar('epoch_' + key, avg_value, EPOCH_CNT)
        log_string('mean %s: %f' % (key, avg_value))
    
    avg_loss = stat_dict['loss/overall_loss'] / num_batches if 'loss/overall_loss' in stat_dict else 0
    net.train()
    return avg_loss


def train(start_epoch, net, optimizer, scaler, device, train_dataloader, val_dataloader, train_writer, val_writer):
    global EPOCH_CNT
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % epoch)
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string(str(datetime.now()))
        
        # Log learning rate to TensorBoard
        train_writer.add_scalar('learning_rate', get_current_lr(epoch), epoch)
        
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_loss = train_one_epoch(net, optimizer, scaler, device, train_dataloader, train_writer)
        
        """# Run validation TODO: currently disabled for faster testing
        log_string('\n---- Running Validation ----')
        val_loss = validate_one_epoch(net, device, val_dataloader, val_writer)
        log_string('Validation Loss: %.4f' % val_loss)
        log_string('Training Loss: %.4f' % train_loss)
        
        # Log train vs val comparison to both writers
        train_writer.add_scalars('epoch_loss_comparison', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        val_writer.add_scalars('epoch_loss_comparison', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_save_path = os.path.join(cfgs.log_dir, cfgs.model_name + '_best.tar')
            save_dict = {'epoch': epoch + 1, 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_state_dict': net.state_dict(),
                        'val_loss': val_loss,
                        'train_loss': train_loss}
            torch.save(save_dict, best_save_path)
            log_string('**** Saved best model with val_loss: %.4f (train_loss: %.4f) ****' % (val_loss, train_loss))
    """
        # Save regular checkpoint
        save_dict = {'epoch': epoch + 1, 'optimizer_state_dict': optimizer.state_dict(),
                     'model_state_dict': net.state_dict(),
                     'scaler_state_dict': scaler.state_dict()}
        torch.save(save_dict, os.path.join(cfgs.log_dir, cfgs.model_name + '_epoch' + str(epoch + 1).zfill(2) + '.tar'))


if __name__ == '__main__':
    # Create dataloaders (only in main process)
    TRAIN_DATALOADER, VAL_DATALOADER = create_dataloaders()
    
    # Create model, optimizer, and scaler (only in main process)
    net, optimizer, scaler, start_epoch, device = create_model_and_optimizer()
    
    # TensorBoard Visualizers
    TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))
    VAL_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'val'))
    
    # Start training
    train(start_epoch, net, optimizer, scaler, device, TRAIN_DATALOADER, VAL_DATALOADER, TRAIN_WRITER, VAL_WRITER)
