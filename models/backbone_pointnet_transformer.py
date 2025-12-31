"""
PointNet Transformer Backbone for Grasp Detection

A pure PyTorch implementation of a PointNet-style Transformer backbone that processes
sparse voxelized point clouds. Designed as a drop-in replacement for backbone_resunet14.

Architecture Overview:
1. Point-wise feature embedding (MLP)
2. Local feature aggregation via k-NN attention
3. Global Transformer layers for long-range dependencies  
4. Multi-scale feature fusion via hierarchical pooling
5. Feature propagation back to original voxel resolution

Key Design Decisions:
- Uses sparse tensor format (spconv.SparseConvTensor) for I/O compatibility
- Internally processes as dense point sets with batch handling
- k-NN based local attention since point clouds lack grid structure
- Relative positional encoding using 3D coordinates
- No external dependencies beyond PyTorch and spconv (for tensor format only)

Hyperparameters tuned for:
- Voxel size: 5mm
- Typical input: 10k-30k voxels per batch after voxelization  
- Scene scale: ~1.2m x 1.2m workspace
- Output: 512-dim features per voxel

Author: Auto-generated for GraspNet pipeline
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv


# =============================================================================
# Utility Functions
# =============================================================================

def knn_query(pos: torch.Tensor, k: int, batch: torch.Tensor = None) -> torch.Tensor:
    """
    Find k-nearest neighbors for each point using pure PyTorch.
    
    Args:
        pos: (N, 3) point positions
        k: number of neighbors
        batch: (N,) batch indices for each point (optional, for batched processing)
    
    Returns:
        idx: (N, k) indices of k-nearest neighbors for each point
    """
    if batch is None:
        # Single batch - compute pairwise distances
        # Use chunked computation for memory efficiency
        N = pos.shape[0]
        k = min(k, N)
        
        if N <= 8192:
            # Small enough to compute full distance matrix
            dist = torch.cdist(pos, pos)  # (N, N)
            _, idx = dist.topk(k, dim=-1, largest=False)  # (N, k)
        else:
            # Chunked computation for large point clouds
            idx = torch.zeros(N, k, dtype=torch.long, device=pos.device)
            chunk_size = 4096
            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                dist_chunk = torch.cdist(pos[i:end_i], pos)  # (chunk, N)
                _, idx[i:end_i] = dist_chunk.topk(k, dim=-1, largest=False)
        return idx
    else:
        # Batched processing - find neighbors within same batch only
        device = pos.device
        N = pos.shape[0]
        k = min(k, N)
        
        # Get unique batch indices
        unique_batches = torch.unique(batch)
        idx = torch.zeros(N, k, dtype=torch.long, device=device)
        
        for b in unique_batches:
            mask = (batch == b)
            pos_b = pos[mask]
            indices_b = torch.where(mask)[0]
            n_b = pos_b.shape[0]
            k_b = min(k, n_b)
            
            if n_b <= 8192:
                dist_b = torch.cdist(pos_b, pos_b)
                _, local_idx = dist_b.topk(k_b, dim=-1, largest=False)
            else:
                local_idx = torch.zeros(n_b, k_b, dtype=torch.long, device=device)
                chunk_size = 4096
                for i in range(0, n_b, chunk_size):
                    end_i = min(i + chunk_size, n_b)
                    dist_chunk = torch.cdist(pos_b[i:end_i], pos_b)
                    _, local_idx[i:end_i] = dist_chunk.topk(k_b, dim=-1, largest=False)
            
            # Map local indices to global indices
            global_idx = indices_b[local_idx]
            
            # Pad if k_b < k
            if k_b < k:
                pad_idx = global_idx[:, :1].expand(-1, k - k_b)
                global_idx = torch.cat([global_idx, pad_idx], dim=1)
            
            idx[mask] = global_idx
        
        return idx


def farthest_point_sampling(pos: torch.Tensor, num_samples: int, batch: torch.Tensor = None) -> torch.Tensor:
    """
    Farthest point sampling using pure PyTorch.
    
    Args:
        pos: (N, 3) point positions
        num_samples: number of points to sample
        batch: (N,) batch indices
    
    Returns:
        idx: (num_samples,) or list of indices per batch
    """
    device = pos.device
    N = pos.shape[0]
    
    if batch is None:
        # Single batch FPS
        num_samples = min(num_samples, N)
        sampled_idx = torch.zeros(num_samples, dtype=torch.long, device=device)
        distances = torch.full((N,), float('inf'), device=device)
        
        # Start from random point
        farthest = torch.randint(0, N, (1,), device=device).item()
        
        for i in range(num_samples):
            sampled_idx[i] = farthest
            centroid = pos[farthest:farthest+1]
            dist = torch.sum((pos - centroid) ** 2, dim=-1)
            distances = torch.minimum(distances, dist)
            farthest = torch.argmax(distances).item()
        
        return sampled_idx
    else:
        # Batched FPS
        unique_batches = torch.unique(batch)
        all_idx = []
        
        for b in unique_batches:
            mask = (batch == b)
            pos_b = pos[mask]
            indices_b = torch.where(mask)[0]
            n_b = pos_b.shape[0]
            num_samples_b = min(num_samples, n_b)
            
            sampled_local = torch.zeros(num_samples_b, dtype=torch.long, device=device)
            distances = torch.full((n_b,), float('inf'), device=device)
            farthest = torch.randint(0, n_b, (1,), device=device).item()
            
            for i in range(num_samples_b):
                sampled_local[i] = farthest
                centroid = pos_b[farthest:farthest+1]
                dist = torch.sum((pos_b - centroid) ** 2, dim=-1)
                distances = torch.minimum(distances, dist)
                farthest = torch.argmax(distances).item()
            
            all_idx.append(indices_b[sampled_local])
        
        return all_idx


# =============================================================================
# Building Blocks
# =============================================================================

class PointEmbedding(nn.Module):
    """
    Initial point-wise feature embedding using MLPs.
    Maps input features + 3D coordinates to higher dimensional space.
    """
    def __init__(self, in_channels: int, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.feat_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
    
    def forward(self, pos: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: (N, 3) normalized coordinates
            feat: (N, C) input features
        Returns:
            (N, embed_dim) embedded features
        """
        pos_embed = self.coord_mlp(pos)
        feat_embed = self.feat_mlp(feat)
        return self.fuse(torch.cat([pos_embed, feat_embed], dim=-1))


class RelativePositionalEncoding(nn.Module):
    """
    Learnable relative positional encoding based on 3D displacement vectors.
    Used in local attention to encode spatial relationships.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # MLP to encode relative positions
        self.pos_enc = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, num_heads),
        )
    
    def forward(self, pos: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Compute relative positional bias for attention.
        
        Args:
            pos: (N, 3) point positions
            idx: (N, k) neighbor indices
        
        Returns:
            (N, num_heads, k) relative positional bias
        """
        N, k = idx.shape
        
        # Get neighbor positions: (N, k, 3)
        neighbor_pos = pos[idx]
        
        # Compute relative positions: (N, k, 3)
        rel_pos = neighbor_pos - pos.unsqueeze(1)
        
        # Encode: (N, k, num_heads) -> (N, num_heads, k)
        bias = self.pos_enc(rel_pos).permute(0, 2, 1)
        
        return bias


class LocalAttention(nn.Module):
    """
    Local self-attention operating on k-nearest neighbors.
    Includes relative positional encoding for spatial awareness.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, k: int = 16, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k = k
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self.rel_pos = RelativePositionalEncoding(embed_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, embed_dim) point features
            pos: (N, 3) point positions
            idx: (N, k) neighbor indices
        
        Returns:
            (N, embed_dim) updated features
        """
        N, C = x.shape
        k = idx.shape[1]
        
        # Compute Q, K, V
        q = self.q_proj(x).view(N, self.num_heads, self.head_dim)  # (N, H, D)
        k_feat = self.k_proj(x)[idx].view(N, k, self.num_heads, self.head_dim)  # (N, k, H, D)
        v = self.v_proj(x)[idx].view(N, k, self.num_heads, self.head_dim)  # (N, k, H, D)
        
        # Attention scores: (N, H, k)
        attn = torch.einsum('nhd,nkhd->nhk', q, k_feat) * self.scale
        
        # Add relative positional bias
        pos_bias = self.rel_pos(pos, idx)  # (N, H, k)
        attn = attn + pos_bias
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Aggregate: (N, H, D)
        out = torch.einsum('nhk,nkhd->nhd', attn, v)
        out = out.reshape(N, C)
        
        # Output projection with residual
        out = self.o_proj(out)
        out = self.norm(x + out)
        
        return out


class GlobalAttention(nn.Module):
    """
    Global self-attention for long-range dependencies.
    Uses standard multi-head attention on subsampled anchor points.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, embed_dim) features
        
        Returns:
            (N, embed_dim) updated features
        """
        N, C = x.shape
        
        # Self-attention
        qkv = self.qkv(x).view(N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each (N, H, D)
        
        # Scaled dot-product attention
        attn = torch.einsum('nhd,mhd->hnm', q, k) * self.scale  # (H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Aggregate
        out = torch.einsum('hnm,mhd->nhd', attn, v)  # (N, H, D)
        out = out.reshape(N, C)
        out = self.o_proj(out)
        
        # Residual + LayerNorm
        x = self.norm1(x + out)
        
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        
        return x


class PointTransformerBlock(nn.Module):
    """
    Combined local + global attention block.
    Local attention captures fine geometric details.
    Global attention (on subset) captures scene-level context.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, k: int = 16, 
                 use_global: bool = True, global_ratio: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.use_global = use_global
        self.global_ratio = global_ratio
        
        self.local_attn = LocalAttention(embed_dim, num_heads, k, dropout)
        
        if use_global:
            self.global_attn = GlobalAttention(embed_dim, num_heads, dropout)
            self.fuse = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor, idx: torch.Tensor, 
                batch: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (N, embed_dim) features
            pos: (N, 3) positions
            idx: (N, k) neighbor indices
            batch: (N,) batch indices
        """
        # Local attention
        x_local = self.local_attn(x, pos, idx)
        
        if self.use_global:
            # Subsample for global attention (for efficiency)
            N = x.shape[0]
            num_global = max(int(N * self.global_ratio), 64)
            num_global = min(num_global, N)
            
            # Random sampling for global attention anchors
            global_idx = torch.randperm(N, device=x.device)[:num_global]
            x_global_in = x_local[global_idx]
            
            # Global attention on subset
            x_global_out = self.global_attn(x_global_in)
            
            # Propagate back to all points via nearest anchor
            if num_global < N:
                # Find nearest global anchor for each point
                dist = torch.cdist(pos, pos[global_idx])  # (N, num_global)
                nearest = dist.argmin(dim=-1)  # (N,)
                x_global_full = x_global_out[nearest]
            else:
                x_global_full = x_global_out
            
            # Fuse local and global
            x = self.fuse(torch.cat([x_local, x_global_full], dim=-1))
        else:
            x = x_local
        
        return x


class SetAbstraction(nn.Module):
    """
    PointNet++ style set abstraction for hierarchical feature learning.
    Downsamples points and aggregates local features.
    """
    def __init__(self, in_channels: int, out_channels: int, ratio: float = 0.25, k: int = 32):
        super().__init__()
        self.ratio = ratio
        self.k = k
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor = None):
        """
        Args:
            x: (N, C) features
            pos: (N, 3) positions
            batch: (N,) batch indices
        
        Returns:
            x_down: (M, out_channels) downsampled features
            pos_down: (M, 3) downsampled positions  
            batch_down: (M,) batch indices for downsampled points
            upsample_idx: indices for upsampling back to original resolution
        """
        N = x.shape[0]
        device = x.device
        
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)
        
        # Farthest point sampling per batch
        unique_batches = torch.unique(batch)
        sampled_idx_list = []
        
        for b in unique_batches:
            mask = (batch == b)
            n_b = mask.sum().item()
            num_samples = max(int(n_b * self.ratio), 1)
            
            pos_b = pos[mask]
            indices_b = torch.where(mask)[0]
            
            # Simple random sampling (faster than FPS for training)
            if num_samples >= n_b:
                sampled_local = torch.arange(n_b, device=device)
            else:
                sampled_local = torch.randperm(n_b, device=device)[:num_samples]
            
            sampled_idx_list.append(indices_b[sampled_local])
        
        sampled_idx = torch.cat(sampled_idx_list)
        
        # Get downsampled positions and batch indices
        pos_down = pos[sampled_idx]
        batch_down = batch[sampled_idx]
        M = pos_down.shape[0]
        
        # Find k-nearest neighbors in original point cloud for each downsampled point
        # Group points within same batch
        x_down_list = []
        
        for b in unique_batches:
            mask_orig = (batch == b)
            mask_down = (batch_down == b)
            
            pos_orig = pos[mask_orig]
            pos_down_b = pos_down[mask_down]
            x_orig = x[mask_orig]
            
            # Find k nearest original points for each downsampled point
            dist = torch.cdist(pos_down_b, pos_orig)  # (M_b, N_b)
            k_b = min(self.k, pos_orig.shape[0])
            _, knn_idx = dist.topk(k_b, dim=-1, largest=False)  # (M_b, k)
            
            # Gather features and relative positions
            knn_feat = x_orig[knn_idx]  # (M_b, k, C)
            knn_pos = pos_orig[knn_idx]  # (M_b, k, 3)
            rel_pos = knn_pos - pos_down_b.unsqueeze(1)  # (M_b, k, 3)
            
            # Concatenate and process
            knn_input = torch.cat([knn_feat, rel_pos], dim=-1)  # (M_b, k, C+3)
            knn_out = self.mlp(knn_input)  # (M_b, k, out_channels)
            
            # Max pooling over neighbors
            x_down_b = knn_out.max(dim=1)[0]  # (M_b, out_channels)
            x_down_list.append(x_down_b)
        
        x_down = torch.cat(x_down_list, dim=0)
        
        # Store indices for upsampling (nearest neighbor interpolation)
        # For each original point, find nearest downsampled point
        upsample_idx = torch.zeros(N, dtype=torch.long, device=device)
        
        for b in unique_batches:
            mask_orig = (batch == b)
            mask_down = (batch_down == b)
            
            pos_orig = pos[mask_orig]
            pos_down_b = pos_down[mask_down]
            
            indices_orig = torch.where(mask_orig)[0]
            indices_down = torch.where(mask_down)[0]
            
            dist = torch.cdist(pos_orig, pos_down_b)
            nearest = dist.argmin(dim=-1)
            upsample_idx[indices_orig] = indices_down[nearest]
        
        return x_down, pos_down, batch_down, upsample_idx


class FeaturePropagation(nn.Module):
    """
    Feature propagation for upsampling features back to original resolution.
    Uses skip connections from encoder.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + skip_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )
    
    def forward(self, x_down: torch.Tensor, x_skip: torch.Tensor, 
                upsample_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_down: (M, C) downsampled features
            x_skip: (N, C_skip) skip connection features from encoder
            upsample_idx: (N,) indices into x_down for each original point
        
        Returns:
            (N, out_channels) upsampled features
        """
        # Interpolate (nearest neighbor)
        x_interp = x_down[upsample_idx]  # (N, C)
        
        # Concatenate with skip connection
        x_cat = torch.cat([x_interp, x_skip], dim=-1)  # (N, C + C_skip)
        
        return self.mlp(x_cat)


# =============================================================================
# Main Backbone
# =============================================================================

class PointNetTransformerBackbone(nn.Module):
    """
    PointNet Transformer Backbone for sparse voxelized point clouds.
    
    Architecture:
    1. Point embedding (coordinate + feature encoding)
    2. Encoder: hierarchical point transformer blocks with set abstraction
    3. Decoder: feature propagation with skip connections
    4. Output MLP
    
    Designed for:
    - Voxel size: 5mm
    - Input: 10k-30k voxels per batch
    - Output: 512-dim features per voxel
    
    Args:
        in_channels: Input feature dimension (default: 3)
        out_channels: Output feature dimension (default: 512)
        embed_dim: Transformer embedding dimension (default: 128)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers per scale (default: 2)
        k_neighbors: Number of neighbors for local attention (default: 16)
        dropout: Dropout rate (default: 0.1)
        D: Spatial dimensions (unused, for API compatibility)
    """
    
    # Hyperparameters exposed as class attributes for easy modification
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = (2, 2, 2)  # Layers per encoder scale
    K_NEIGHBORS = 16
    DOWNSAMPLE_RATIOS = (0.5, 0.5)  # Hierarchical downsampling ratios
    DECODER_DIMS = (256, 192)  # Feature propagation output dims
    
    def __init__(self, in_channels: int = 3, out_channels: int = 512, D: int = 3,
                 embed_dim: int = None, num_heads: int = None, num_layers: tuple = None,
                 k_neighbors: int = None, dropout: float = 0.1):
        super().__init__()
        
        # Use class attributes as defaults, allow override
        embed_dim = embed_dim or self.EMBED_DIM
        num_heads = num_heads or self.NUM_HEADS
        num_layers = num_layers or self.NUM_LAYERS
        k_neighbors = k_neighbors or self.K_NEIGHBORS
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.k_neighbors = k_neighbors
        
        # Encoder dimensions: progressively increase
        enc_dims = [embed_dim, embed_dim * 2, embed_dim * 4]
        
        # =========================
        # Encoder
        # =========================
        
        # Initial embedding
        self.embedding = PointEmbedding(in_channels, enc_dims[0], hidden_dim=64)
        
        # Scale 0 (full resolution)
        self.enc_blocks_0 = nn.ModuleList([
            PointTransformerBlock(enc_dims[0], num_heads, k_neighbors, 
                                  use_global=False, dropout=dropout)
            for _ in range(num_layers[0])
        ])
        
        # Downsample to scale 1
        self.down_0 = SetAbstraction(enc_dims[0], enc_dims[1], 
                                     ratio=self.DOWNSAMPLE_RATIOS[0], k=k_neighbors * 2)
        
        # Scale 1
        self.enc_blocks_1 = nn.ModuleList([
            PointTransformerBlock(enc_dims[1], num_heads, k_neighbors,
                                  use_global=True, global_ratio=0.2, dropout=dropout)
            for _ in range(num_layers[1])
        ])
        
        # Downsample to scale 2
        self.down_1 = SetAbstraction(enc_dims[1], enc_dims[2],
                                     ratio=self.DOWNSAMPLE_RATIOS[1], k=k_neighbors * 2)
        
        # Scale 2 (bottleneck - lowest resolution)
        self.enc_blocks_2 = nn.ModuleList([
            PointTransformerBlock(enc_dims[2], num_heads, k_neighbors,
                                  use_global=True, global_ratio=0.5, dropout=dropout)
            for _ in range(num_layers[2])
        ])
        
        # =========================
        # Decoder
        # =========================
        
        dec_dims = self.DECODER_DIMS
        
        # Upsample from scale 2 to scale 1
        self.up_2 = FeaturePropagation(enc_dims[2], enc_dims[1], dec_dims[0])
        
        # Upsample from scale 1 to scale 0
        self.up_1 = FeaturePropagation(dec_dims[0], enc_dims[0], dec_dims[1])
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(dec_dims[1], out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """
        Forward pass compatible with spconv SparseConvTensor.
        
        Args:
            x: spconv.SparseConvTensor with:
                - features: (M, in_channels) voxel features
                - indices: (M, 4) voxel coordinates [batch, x, y, z]
                - spatial_shape: (X, Y, Z)
                - batch_size: B
        
        Returns:
            spconv.SparseConvTensor with same structure but features: (M, out_channels)
        """
        # Extract data from sparse tensor
        feats = x.features  # (M, in_channels)
        coords = x.indices  # (M, 4) [batch, x, y, z]
        spatial_shape = x.spatial_shape
        batch_size = x.batch_size
        
        # Convert voxel coordinates to normalized positions
        # Normalize by spatial shape for position encoding
        pos = coords[:, 1:].float()  # (M, 3) [x, y, z]
        spatial_tensor = torch.tensor(spatial_shape, dtype=torch.float32, device=pos.device)
        pos = pos / spatial_tensor.clamp(min=1)  # Normalize to [0, 1]
        
        # Batch indices
        batch = coords[:, 0]  # (M,)
        
        M = feats.shape[0]
        device = feats.device
        
        # =========================
        # Encoder
        # =========================
        
        # Initial embedding
        x0 = self.embedding(pos, feats)  # (M, embed_dim)
        
        # Compute k-NN indices for scale 0
        k0 = min(self.k_neighbors, M)
        idx_0 = knn_query(pos, k0, batch)
        
        # Scale 0 transformer blocks
        for block in self.enc_blocks_0:
            x0 = block(x0, pos, idx_0, batch)
        
        # Downsample to scale 1
        x1, pos1, batch1, up_idx_1 = self.down_0(x0, pos, batch)
        
        # Compute k-NN for scale 1
        M1 = x1.shape[0]
        k1 = min(self.k_neighbors, M1)
        idx_1 = knn_query(pos1, k1, batch1)
        
        # Scale 1 transformer blocks
        for block in self.enc_blocks_1:
            x1 = block(x1, pos1, idx_1, batch1)
        
        # Downsample to scale 2
        x2, pos2, batch2, up_idx_2 = self.down_1(x1, pos1, batch1)
        
        # Compute k-NN for scale 2
        M2 = x2.shape[0]
        k2 = min(self.k_neighbors, M2)
        idx_2 = knn_query(pos2, k2, batch2)
        
        # Scale 2 transformer blocks (bottleneck)
        for block in self.enc_blocks_2:
            x2 = block(x2, pos2, idx_2, batch2)
        
        # =========================
        # Decoder
        # =========================
        
        # Upsample scale 2 -> scale 1
        x1_up = self.up_2(x2, x1, up_idx_2)
        
        # Upsample scale 1 -> scale 0
        x0_up = self.up_1(x1_up, x0, up_idx_1)
        
        # Output MLP
        out_feats = self.output_mlp(x0_up)  # (M, out_channels)
        
        # Create output sparse tensor with same structure
        out = spconv.SparseConvTensor(
            out_feats,
            coords,
            spatial_shape,
            batch_size
        )
        
        return out


# =============================================================================
# Variant Configurations (matching UNet naming convention)
# =============================================================================

class PointNetTransformer14D(PointNetTransformerBackbone):
    """
    Default configuration matching SPconvUNet14D output dimensions.
    ~14M parameters, suitable for grasp detection.
    """
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = (2, 2, 2)
    K_NEIGHBORS = 16
    DOWNSAMPLE_RATIOS = (0.5, 0.5)
    DECODER_DIMS = (256, 192)


class PointNetTransformerSmall(PointNetTransformerBackbone):
    """
    Smaller variant for faster training/inference.
    ~5M parameters.
    """
    EMBED_DIM = 96
    NUM_HEADS = 6
    NUM_LAYERS = (1, 2, 1)
    K_NEIGHBORS = 12
    DOWNSAMPLE_RATIOS = (0.5, 0.5)
    DECODER_DIMS = (192, 128)


class PointNetTransformerLarge(PointNetTransformerBackbone):
    """
    Larger variant for potentially better accuracy.
    ~30M parameters.
    """
    EMBED_DIM = 192
    NUM_HEADS = 12
    NUM_LAYERS = (2, 3, 2)
    K_NEIGHBORS = 24
    DOWNSAMPLE_RATIOS = (0.5, 0.5)
    DECODER_DIMS = (384, 256)


class PointNetTransformerTiny(PointNetTransformerBackbone):
    """
    Minimal variant for debugging and quick experiments.
    ~2M parameters.
    """
    EMBED_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = (1, 1, 1)
    K_NEIGHBORS = 8
    DOWNSAMPLE_RATIOS = (0.5, 0.5)
    DECODER_DIMS = (128, 96)


# =============================================================================
# Testing / Validation
# =============================================================================

if __name__ == "__main__":
    """
    Test the backbone with synthetic data to verify shapes and functionality.
    """
    import time
    
    print("=" * 60)
    print("PointNet Transformer Backbone Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Simulate typical input
    B = 2  # batch size
    M = 8000  # number of voxels (typical after voxelization)
    in_channels = 3
    out_channels = 512
    
    # Create synthetic sparse tensor
    # Coordinates: [batch, x, y, z]
    coords = torch.zeros(M, 4, dtype=torch.int32, device=device)
    coords[:, 0] = torch.randint(0, B, (M,))  # batch index
    coords[:, 1] = torch.randint(0, 200, (M,))  # x
    coords[:, 2] = torch.randint(0, 200, (M,))  # y
    coords[:, 3] = torch.randint(0, 80, (M,))   # z
    
    # Make coordinates unique
    coords, _ = torch.unique(coords, dim=0, return_inverse=True)
    M = coords.shape[0]
    
    feats = torch.ones(M, in_channels, dtype=torch.float32, device=device)
    spatial_shape = (200, 200, 80)
    
    sparse_input = spconv.SparseConvTensor(
        feats, coords, spatial_shape, B
    )
    
    print(f"\nInput shape: feats={feats.shape}, coords={coords.shape}")
    print(f"Spatial shape: {spatial_shape}")
    print(f"Batch size: {B}")
    
    # Test each variant
    variants = [
        ("Tiny", PointNetTransformerTiny),
        ("Small", PointNetTransformerSmall),
        ("14D (Default)", PointNetTransformer14D),
        ("Large", PointNetTransformerLarge),
    ]
    
    for name, ModelClass in variants:
        print(f"\n{'-' * 40}")
        print(f"Testing {name} variant...")
        
        model = ModelClass(in_channels=in_channels, out_channels=out_channels).to(device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params / 1e6:.2f}M")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            start = time.time()
            output = model(sparse_input)
            torch.cuda.synchronize() if device.type == "cuda" else None
            elapsed = time.time() - start
        
        print(f"Output features shape: {output.features.shape}")
        print(f"Output coords shape: {output.indices.shape}")
        print(f"Forward time: {elapsed * 1000:.1f}ms")
        
        # Verify output
        assert output.features.shape == (M, out_channels), f"Expected ({M}, {out_channels})"
        assert torch.equal(output.indices, coords), "Coordinates should be preserved"
        print("✓ Shape verification passed")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
