import os
import sys
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.graspnet import GraspNet
from models.loss import get_loss
from dataset.graspnet_dataset import GraspNetDataset, spconv_collate_fn, load_grasp_labels

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
parser.add_argument('--num_workers', type=int, default=12, help='Number of dataloader workers [default: 12]')
parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='Gradient accumulation steps [default: 1]')
parser.add_argument('--use_amp', action='store_true', default=False, help='Use automatic mixed precision training')
parser.add_argument('--resume', action='store_true', default=False, help='Whether to resume from checkpoint')
parser.add_argument('--val_split', type=str, default='val', choices=['val', 'test_seen'], 
                    help='Validation split: "val" uses scenes 7-8 (has labels), "test_seen" uses scene 10 (needs label generation) [default: val]')
cfgs = parser.parse_args()
# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
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


grasp_labels = load_grasp_labels(cfgs.dataset_root)
TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, grasp_labels=grasp_labels, camera=cfgs.camera, split='train',
                                num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                remove_outlier=True, augment=True, load_label=True)
print('train dataset length: ', len(TRAIN_DATASET))

# Validation dataset (use specified validation split without augmentation)
VAL_DATASET = GraspNetDataset(cfgs.dataset_root, grasp_labels=grasp_labels, camera=cfgs.camera, split=cfgs.val_split,
                              num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                              remove_outlier=True, augment=False, load_label=True)
print('validation dataset length: ', len(VAL_DATASET))

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
                              num_workers=cfgs.num_workers, worker_init_fn=my_worker_init_fn, collate_fn=spconv_collate_fn, 
                              pin_memory=True, persistent_workers=True, prefetch_factor=2)
print('train dataloader length: ', len(TRAIN_DATALOADER))

VAL_DATALOADER = DataLoader(VAL_DATASET, batch_size=cfgs.batch_size, shuffle=False,
                            num_workers=cfgs.num_workers, worker_init_fn=my_worker_init_fn, collate_fn=spconv_collate_fn, 
                            pin_memory=True, persistent_workers=True, prefetch_factor=2)
print('validation dataloader length: ', len(VAL_DATALOADER))

net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate)

# Create gradient scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=cfgs.use_amp)

start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))
# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))
VAL_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'val'))


def get_current_lr(epoch):
    lr = cfgs.learning_rate
    lr = lr * (0.95 ** epoch)
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch():
    stat_dict = {}  # collect statistics
    epoch_stat_dict = {}  # collect epoch-level statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()
    batch_interval = 50  # Reduced logging frequency to minimize overhead
    
    for batch_idx, batch_data_label in tqdm(enumerate(TRAIN_DATALOADER), desc='Training'):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)

        # Use automatic mixed precision if enabled
        with torch.cuda.amp.autocast(enabled=cfgs.use_amp):
            end_points = net(batch_data_label)
            loss, end_points = get_loss(end_points)
            loss = loss / cfgs.grad_accumulation_steps  # Scale loss for gradient accumulation
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Update weights only after accumulating gradients
        if (batch_idx + 1) % cfgs.grad_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

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
                TRAIN_WRITER.add_scalar(key, stat_dict[key] / batch_interval,
                                        (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size)
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0
    
    # Log epoch-level averages to TensorBoard
    num_batches = len(TRAIN_DATALOADER)
    for key in sorted(epoch_stat_dict.keys()):
        avg_value = epoch_stat_dict[key] / num_batches
        TRAIN_WRITER.add_scalar('epoch_' + key, avg_value, EPOCH_CNT)
    
    # Return epoch average loss
    return epoch_stat_dict['loss/overall_loss'] / num_batches if 'loss/overall_loss' in epoch_stat_dict else 0


def validate_one_epoch():
    """Run validation and return average loss and statistics"""
    stat_dict = {}  # collect statistics
    net.eval()
    
    with torch.no_grad():
        for batch_idx, batch_data_label in tqdm(enumerate(VAL_DATALOADER), desc='Validating'):
            for key in batch_data_label:
                if 'list' in key:
                    for i in range(len(batch_data_label[key])):
                        for j in range(len(batch_data_label[key][i])):
                            batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
                else:
                    batch_data_label[key] = batch_data_label[key].to(device)

            # Use AMP for validation too
            with torch.cuda.amp.autocast(enabled=cfgs.use_amp):
                end_points = net(batch_data_label)
                loss, end_points = get_loss(end_points)

            for key in end_points:
                if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                    if key not in stat_dict:
                        stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()
    
    # Calculate averages and log to TensorBoard
    num_batches = len(VAL_DATALOADER)
    log_string('---- Validation Results ----')
    for key in sorted(stat_dict.keys()):
        avg_value = stat_dict[key] / num_batches
        # Log with 'epoch_' prefix to match training metrics for comparison
        VAL_WRITER.add_scalar('epoch_' + key, avg_value, EPOCH_CNT)
        log_string('mean %s: %f' % (key, avg_value))
    
    avg_loss = stat_dict['loss/overall_loss'] / num_batches if 'loss/overall_loss' in stat_dict else 0
    net.train()
    return avg_loss


def train(start_epoch):
    global EPOCH_CNT
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % epoch)
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string(str(datetime.now()))
        
        # Log learning rate to TensorBoard
        TRAIN_WRITER.add_scalar('learning_rate', get_current_lr(epoch), epoch)
        
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_loss = train_one_epoch()
        
        # Run validation
        log_string('\n---- Running Validation ----')
        val_loss = validate_one_epoch()
        log_string('Validation Loss: %.4f' % val_loss)
        log_string('Training Loss: %.4f' % train_loss)
        
        # Log train vs val comparison to both writers
        TRAIN_WRITER.add_scalars('epoch_loss_comparison', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        VAL_WRITER.add_scalars('epoch_loss_comparison', {
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

        # Save regular checkpoint
        save_dict = {'epoch': epoch + 1, 'optimizer_state_dict': optimizer.state_dict(),
                     'model_state_dict': net.state_dict()}
        torch.save(save_dict, os.path.join(cfgs.log_dir, cfgs.model_name + '_epoch' + str(epoch + 1).zfill(2) + '.tar'))


if __name__ == '__main__':
    train(start_epoch)
