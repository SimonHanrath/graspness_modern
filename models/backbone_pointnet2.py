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
    PointNet++ encoder-decoder backbone (enhanced version).
    
    Takes raw point cloud via forward_points() [preferred], or sparse voxel 
    tensor via forward() [legacy, for backwards compatibility].
    
    Architecture improvements over original:
    - 4 SA layers (vs 3) for deeper feature learning matching ResUNet depth
    - Wider MLPs (64/128 base channels vs 32/64) for more capacity
    - Larger radii progression to capture global scene context
    
    Note: The SA layers perform their own internal FPS subsampling 
    (15000 -> 1024 -> 256 -> 64 -> 16), which is the standard PointNet++ design.
    All input points receive features via the FP (feature propagation) decoder.
    """
    
    def __init__(self, in_channels=3, out_channels=512, max_input_points=8192):
        super().__init__()
        self.out_channels = out_channels
        self.max_input_points = max_input_points  # Only used in legacy forward() path
        
        # Encoder - 4 SA layers with wider MLPs
        # Subsampling: 8192 -> 1024 -> 256 -> 64 -> 16
        self.sa1 = PointnetSAModuleMSG(
            npoint=1024,
            radii=[0.02, 0.04],
            nsamples=[32, 32],
            mlps=[[in_channels, 64, 128], [in_channels, 64, 128]],  # Doubled channels
            use_xyz=True
        )
        
        self.sa2 = PointnetSAModuleMSG(
            npoint=256,
            radii=[0.04, 0.08],
            nsamples=[32, 32],
            mlps=[[256, 128, 256], [256, 128, 256]],  # Doubled channels
            use_xyz=True
        )
        
        self.sa3 = PointnetSAModuleMSG(
            npoint=64,
            radii=[0.08, 0.16],
            nsamples=[32, 32],
            mlps=[[512, 256, 512], [512, 256, 512]],  # Doubled channels
            use_xyz=True
        )
        
        # SA4 bottleneck - captures global scene context
        self.sa4 = PointnetSAModuleMSG(
            npoint=16,
            radii=[0.16, 0.32],
            nsamples=[32, 32],
            mlps=[[1024, 512, 512], [1024, 512, 512]],
            use_xyz=True
        )
        
        # Decoder - 4 FP layers propagating features back to input resolution
        self.fp4 = PointnetFPModule(mlp=[1024 + 1024, 512, 512])  # sa4 + sa3 features
        self.fp3 = PointnetFPModule(mlp=[512 + 512, 512, 512])    # fp4 + sa2 features
        self.fp2 = PointnetFPModule(mlp=[512 + 256, 256, 256])    # fp3 + sa1 features
        self.fp1 = PointnetFPModule(mlp=[256 + in_channels, 256, 256])  # fp2 + input features
        
        # Final projection after interpolation to all points
        # NOTE: No ReLU here! ResUNet doesn't use final activation,
        # allowing negative features for better discriminability in GraspableNet
        self.final_mlp = nn.Sequential(
            nn.Conv1d(256, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward_points(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on raw point cloud (NOT voxelized).
        
        This is the preferred interface for PointNet++ - it operates on actual
        point positions rather than voxel centers, preserving sub-voxel geometry.
        
        All input points are processed directly through the network (no FPS 
        pre-subsampling). The SA layers handle their own internal subsampling
        via FPS, which is the standard PointNet++ design.
        
        Args:
            xyz: (B, N, 3) point coordinates in meters
            feats: (B, N, C) per-point features (e.g., RGB or ones)
            
        Returns:
            (B, N, out_channels) per-point features
        """
        # PointNet++ expects features as (B, C, N)
        feats = feats.transpose(1, 2)
        
        # Encoder - 4 SA layers (each does its own FPS internally)
        # 15000 -> 1024 -> 256 -> 64 -> 16
        l0_xyz, l0_feats = xyz, feats
        l1_xyz, l1_feats = self.sa1(l0_xyz, l0_feats)
        l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats)
        l3_xyz, l3_feats = self.sa3(l2_xyz, l2_feats)
        l4_xyz, l4_feats = self.sa4(l3_xyz, l3_feats)
        
        # Decoder - 4 FP layers (propagate features back to full resolution)
        # 16 -> 64 -> 256 -> 1024 -> 15000
        l3_feats = self.fp4(l3_xyz, l4_xyz, l3_feats, l4_feats)
        l2_feats = self.fp3(l2_xyz, l3_xyz, l2_feats, l3_feats)
        l1_feats = self.fp2(l1_xyz, l2_xyz, l1_feats, l2_feats)
        l0_feats = self.fp1(l0_xyz, l1_xyz, l0_feats, l1_feats)
        
        # Final projection
        out_feats = self.final_mlp(l0_feats)  # (B, out_channels, N)
        return out_feats.transpose(1, 2)  # (B, N, out_channels)
    
    def forward(self, sparse_input: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        feats = sparse_input.features  # (M, C)
        coords = sparse_input.indices  # (M, 4) [batch, x, y, z]
        batch_size = sparse_input.batch_size
        
        # CRITICAL: coords are voxel indices. PointNet++ needs real spatial coordinates!
        # Dataset uses voxel_size=0.005, so multiply back to get meters
        VOXEL_SIZE = 0.005
        xyz = coords[:, 1:4].float() * VOXEL_SIZE
        batch_idx = coords[:, 0].int()
        
        # OPTIMIZATION: Fast path for batch_size=1 (skip padding overhead)
        if batch_size == 1:
            xyz_batched = xyz.unsqueeze(0)  # (1, N, 3)
            feats_batched = feats.unsqueeze(0)  # (1, N, C)
            points_per_batch = [xyz.shape[0]]
        else:
            # Count points per batch (vectorized)
            points_per_batch = torch.bincount(batch_idx, minlength=batch_size).tolist()
            max_points = max(points_per_batch)
            
            # Vectorized batching using scatter
            xyz_batched = torch.zeros(batch_size, max_points, 3, device=xyz.device, dtype=xyz.dtype)
            feats_batched = torch.zeros(batch_size, max_points, feats.shape[1], device=feats.device, dtype=feats.dtype)
            
            # Scatter operation is faster than manual loop
            batch_idx_expanded = batch_idx.unsqueeze(1).expand(-1, 3)  # (M, 3)
            xyz_batched.scatter_(1, batch_idx_expanded.unsqueeze(2).expand(-1, -1, 1), xyz.unsqueeze(1))
            
            # For features, need to handle variable feature dimensions
            for b in range(batch_size):
                mask = batch_idx == b
                n = mask.sum().item()
                xyz_batched[b, :n] = xyz[mask]
                feats_batched[b, :n] = feats[mask]
        
        # Subsample to max_input_points using FPS for memory efficiency
        max_points = max(points_per_batch)
        if max_points > self.max_input_points:
            fps_idx = furthest_point_sample(xyz_batched, self.max_input_points)
            xyz_sub = gather_operation(xyz_batched.transpose(1, 2), fps_idx).transpose(1, 2)
            feats_sub = gather_operation(feats_batched.transpose(1, 2), fps_idx).transpose(1, 2)
        else:
            xyz_sub = xyz_batched
            feats_sub = feats_batched
            fps_idx = None
        
        # PointNet++ expects features as (B, C, N)
        feats_sub = feats_sub.transpose(1, 2)
        
        # Encoder - 4 SA layers
        l0_xyz, l0_feats = xyz_sub, feats_sub
        l1_xyz, l1_feats = self.sa1(l0_xyz, l0_feats)   # 8192 -> 1024
        l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats)   # 1024 -> 256
        l3_xyz, l3_feats = self.sa3(l2_xyz, l2_feats)   # 256 -> 64
        l4_xyz, l4_feats = self.sa4(l3_xyz, l3_feats)   # 64 -> 16 (bottleneck)
        
        # Decoder - 4 FP layers back to subsampled resolution
        l3_feats = self.fp4(l3_xyz, l4_xyz, l3_feats, l4_feats)  # 16 -> 64
        l2_feats = self.fp3(l2_xyz, l3_xyz, l2_feats, l3_feats)  # 64 -> 256
        l1_feats = self.fp2(l1_xyz, l2_xyz, l1_feats, l2_feats)  # 256 -> 1024
        l0_feats = self.fp1(l0_xyz, l1_xyz, l0_feats, l1_feats)  # 1024 -> 8192 (B, 256, max_input_points)
        
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
        
        # Unpad and concatenate (vectorized for batch > 1, simple squeeze for batch_size=1)
        if batch_size == 1:
            out_feats = out_feats.squeeze(0)
        else:
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
        
        # NOTE: No ReLU here! ResUNet doesn't use final activation,
        # allowing negative features for better discriminability in GraspableNet
        self.final_mlp = nn.Sequential(
            nn.Conv1d(128, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward(self, sparse_input: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        feats = sparse_input.features
        coords = sparse_input.indices
        batch_size = sparse_input.batch_size
        
        # CRITICAL: coords are voxel indices. PointNet++ needs real spatial coordinates!
        VOXEL_SIZE = 0.005 # TODO: Do not hardcode this
        xyz = coords[:, 1:4].float() * VOXEL_SIZE
        batch_idx = coords[:, 0].int()
        
        # OPTIMIZATION: Fast path for batch_size=1 (skip padding overhead)
        if batch_size == 1:
            xyz_batched = xyz.unsqueeze(0)  # (1, N, 3)
            feats_batched = feats.unsqueeze(0)  # (1, N, C)
            points_per_batch = [xyz.shape[0]]
        else:
            # Count points per batch (vectorized)
            points_per_batch = torch.bincount(batch_idx, minlength=batch_size).tolist()
            max_points = max(points_per_batch)
            
            xyz_batched = torch.zeros(batch_size, max_points, 3, device=xyz.device, dtype=xyz.dtype)
            feats_batched = torch.zeros(batch_size, max_points, feats.shape[1], device=feats.device, dtype=feats.dtype)
            
            for b in range(batch_size):
                mask = batch_idx == b
                n = mask.sum().item()
                xyz_batched[b, :n] = xyz[mask]
                feats_batched[b, :n] = feats[mask]
        
        # Subsample if needed
        max_points = max(points_per_batch)
        if max_points > self.max_input_points:
            fps_idx = furthest_point_sample(xyz_batched, self.max_input_points)
            xyz_sub = gather_operation(xyz_batched.transpose(1, 2), fps_idx).transpose(1, 2)
            feats_sub = gather_operation(feats_batched.transpose(1, 2), fps_idx).transpose(1, 2)
        else:
            xyz_sub = xyz_batched
            feats_sub = feats_batched
            fps_idx = None
        
        feats_sub = feats_sub.transpose(1, 2)
        
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
        
        # Unpad and concatenate
        if batch_size == 1:
            out_feats = out_feats.squeeze(0)
        else:
            out_feats = torch.cat([out_feats[b, :points_per_batch[b]] for b in range(batch_size)], dim=0)
        
        return sparse_input.replace_feature(out_feats)


# Alias for consistency with other backbone naming
PointNet2D = PointNet2Backbone
PointNet2DLight = PointNet2BackboneLight
