"""
PointNet++ Backbone for Grasp Detection

Pure PyTorch implementation using existing pointnet2 modules.
Follows the same spconv tensor interface as other backbones for drop-in replacement.
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from torch.utils.checkpoint import checkpoint

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from utils.pointnet.pointnet2_modules import PointnetSAModuleMSG, PointnetFPModule
from utils.pointnet.pointnet2_utils import furthest_point_sample, gather_operation, three_nn, three_interpolate


class PointNet2Backbone(nn.Module):
    """
    PointNet++ encoder-decoder backbone.
    
    Takes sparse voxel tensor, processes internally as point cloud,
    returns sparse tensor with same coordinates and output features.
    
    Uses FPS to subsample input to max_input_points before processing,
    then interpolates features back to all original points.
    """
    
    def __init__(self, in_channels=3, out_channels=512, max_input_points=2048):
        super().__init__()
        self.out_channels = out_channels
        self.max_input_points = max_input_points
        
        # Encoder - process subsampled points
        self.sa1 = PointnetSAModuleMSG(
            npoint=512,
            radii=[0.02, 0.04],
            nsamples=[16, 16],
            mlps=[[in_channels, 32, 64], [in_channels, 32, 64]],
            use_xyz=True
        )
        
        self.sa2 = PointnetSAModuleMSG(
            npoint=128,
            radii=[0.04, 0.08],
            nsamples=[16, 16],
            mlps=[[128, 64, 128], [128, 64, 128]],
            use_xyz=True
        )
        
        self.sa3 = PointnetSAModuleMSG(
            npoint=32,
            radii=[0.08, 0.16],
            nsamples=[16, 16],
            mlps=[[256, 128, 256], [256, 128, 256]],
            use_xyz=True
        )
        
        # Decoder - propagate back to max_input_points resolution
        self.fp3 = PointnetFPModule(mlp=[512 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 128, 256, 256])
        self.fp1 = PointnetFPModule(mlp=[256 + in_channels, 256, 256])
        
        # Final projection after interpolation to all points
        self.final_mlp = nn.Sequential(
            nn.Conv1d(256, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, sparse_input: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        feats = sparse_input.features  # (M, C)
        coords = sparse_input.indices  # (M, 4) [batch, x, y, z]
        batch_size = sparse_input.batch_size
        
        xyz = coords[:, 1:4].float()
        batch_idx = coords[:, 0]
        
        # Group by batch
        points_per_batch = [(batch_idx == b).sum().item() for b in range(batch_size)]
        max_points = max(points_per_batch)
        
        # Pad to create batched tensors
        xyz_batched = torch.zeros(batch_size, max_points, 3, device=xyz.device, dtype=xyz.dtype)
        feats_batched = torch.zeros(batch_size, max_points, feats.shape[1], device=feats.device, dtype=feats.dtype)
        
        idx = 0
        for b in range(batch_size):
            n = points_per_batch[b]
            xyz_batched[b, :n] = xyz[idx:idx + n]
            feats_batched[b, :n] = feats[idx:idx + n]
            idx += n
        
        # Subsample to max_input_points using FPS for memory efficiency
        if max_points > self.max_input_points:
            fps_idx = furthest_point_sample(xyz_batched, self.max_input_points)  # (B, max_input_points)
            xyz_sub = gather_operation(xyz_batched.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
            feats_sub = gather_operation(feats_batched.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
        else:
            xyz_sub = xyz_batched
            feats_sub = feats_batched
            fps_idx = None
        
        # PointNet++ expects features as (B, C, N)
        feats_sub = feats_sub.transpose(1, 2).contiguous()
        
        # Encoder
        l0_xyz, l0_feats = xyz_sub, feats_sub
        l1_xyz, l1_feats = self.sa1(l0_xyz, l0_feats)
        l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats)
        l3_xyz, l3_feats = self.sa3(l2_xyz, l2_feats)
        
        # Decoder - back to subsampled resolution
        l2_feats = self.fp3(l2_xyz, l3_xyz, l2_feats, l3_feats)
        l1_feats = self.fp2(l1_xyz, l2_xyz, l1_feats, l2_feats)
        l0_feats = self.fp1(l0_xyz, l1_xyz, l0_feats, l1_feats)  # (B, 256, max_input_points)
        
        # Interpolate back to all original points
        if fps_idx is not None:
            # Use 3-NN interpolation from subsampled to full resolution
            dist, idx_nn = three_nn(xyz_batched, xyz_sub)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = dist_recip.sum(dim=2, keepdim=True)
            weight = dist_recip / norm
            l0_feats = three_interpolate(l0_feats, idx_nn.long(), weight)  # (B, 256, max_points)
        
        # Final projection
        out_feats = self.final_mlp(l0_feats)  # (B, out_channels, N)
        out_feats = out_feats.transpose(1, 2)  # (B, N, C)
        
        # Unpad and concatenate
        out_list = [out_feats[b, :points_per_batch[b]] for b in range(batch_size)]
        out_feats = torch.cat(out_list, dim=0)
        
        return sparse_input.replace_feature(out_feats)


class PointNet2BackboneLight(nn.Module):
    """Ultra-light version for limited GPU memory (<8GB)."""
    
    def __init__(self, in_channels=3, out_channels=512, max_input_points=1024):
        super().__init__()
        self.out_channels = out_channels
        self.max_input_points = max_input_points
        
        # Minimal encoder - single scale for speed
        self.sa1 = PointnetSAModuleMSG(
            npoint=256,
            radii=[0.04],
            nsamples=[16],
            mlps=[[in_channels, 64, 128]],
            use_xyz=True
        )
        
        self.sa2 = PointnetSAModuleMSG(
            npoint=64,
            radii=[0.08],
            nsamples=[16],
            mlps=[[128, 128, 256]],
            use_xyz=True
        )
        
        self.sa3 = PointnetSAModuleMSG(
            npoint=16,
            radii=[0.16],
            nsamples=[16],
            mlps=[[256, 256, 512]],
            use_xyz=True
        )
        
        # Decoder
        self.fp3 = PointnetFPModule(mlp=[512 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 128, 128, 128])
        self.fp1 = PointnetFPModule(mlp=[128 + in_channels, 128, 128])
        
        self.final_mlp = nn.Sequential(
            nn.Conv1d(128, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, sparse_input: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        feats = sparse_input.features
        coords = sparse_input.indices
        batch_size = sparse_input.batch_size
        
        xyz = coords[:, 1:4].float()
        batch_idx = coords[:, 0]
        
        points_per_batch = [(batch_idx == b).sum().item() for b in range(batch_size)]
        max_points = max(points_per_batch)
        
        xyz_batched = torch.zeros(batch_size, max_points, 3, device=xyz.device, dtype=xyz.dtype)
        feats_batched = torch.zeros(batch_size, max_points, feats.shape[1], device=feats.device, dtype=feats.dtype)
        
        idx = 0
        for b in range(batch_size):
            n = points_per_batch[b]
            xyz_batched[b, :n] = xyz[idx:idx + n]
            feats_batched[b, :n] = feats[idx:idx + n]
            idx += n
        
        # Subsample if needed
        if max_points > self.max_input_points:
            fps_idx = furthest_point_sample(xyz_batched, self.max_input_points)
            xyz_sub = gather_operation(xyz_batched.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
            feats_sub = gather_operation(feats_batched.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
        else:
            xyz_sub = xyz_batched
            feats_sub = feats_batched
            fps_idx = None
        
        feats_sub = feats_sub.transpose(1, 2).contiguous()
        
        l0_xyz, l0_feats = xyz_sub, feats_sub
        l1_xyz, l1_feats = self.sa1(l0_xyz, l0_feats)
        l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats)
        l3_xyz, l3_feats = self.sa3(l2_xyz, l2_feats)
        
        l2_feats = self.fp3(l2_xyz, l3_xyz, l2_feats, l3_feats)
        l1_feats = self.fp2(l1_xyz, l2_xyz, l1_feats, l2_feats)
        l0_feats = self.fp1(l0_xyz, l1_xyz, l0_feats, l1_feats)
        
        # Interpolate back to all points
        if fps_idx is not None:
            dist, idx_nn = three_nn(xyz_batched, xyz_sub)
            dist_recip = 1.0 / (dist + 1e-8)
            weight = dist_recip / dist_recip.sum(dim=2, keepdim=True)
            l0_feats = three_interpolate(l0_feats, idx_nn.long(), weight)
        
        out_feats = self.final_mlp(l0_feats).transpose(1, 2)
        out_feats = torch.cat([out_feats[b, :points_per_batch[b]] for b in range(batch_size)], dim=0)
        
        return sparse_input.replace_feature(out_feats)


# Alias for consistency with other backbone naming
PointNet2D = PointNet2Backbone
PointNet2DLight = PointNet2BackboneLight
