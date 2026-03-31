"""
Sonata Backbone for Grasp Detection

Self-supervised pretrained Point Transformer V3 (encoder-only).
Paper: "Sonata: Self-Supervised Learning of Reliable Point Representations" (CVPR 2025 Highlight)
Source: https://github.com/facebookresearch/sonata

This wrapper provides:
- Automatic normal estimation using Open3D (Sonata expects coord + color + normal)
- Conversion from spconv.SparseConvTensor to Sonata Point format
- Feature unpooling back to original resolution
- Output as spconv.SparseConvTensor for compatibility with GraspNet pipeline
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv
from functools import partial
from typing import Optional, Dict, Any

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not available. Normal estimation will use fallback method.")

try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

from .structure import Point
from .modules import PointModule, PointSequential
from .misc import offset2batch, batch2offset


# =============================================================================
# Inline components (same as in backbone_ptv3.py)
# =============================================================================

class DropPath(nn.Module):
    """Stochastic Depth (drop path) regularization."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    """Segment reduction using CSR index pointers."""
    num_segments = indptr.shape[0] - 1
    
    if num_segments == 0:
        return torch.zeros((0, *src.shape[1:]), dtype=src.dtype, device=src.device)
    
    if src.shape[0] == 0:
        return torch.zeros((num_segments, *src.shape[1:]), dtype=src.dtype, device=src.device)
    
    indices = torch.arange(src.shape[0], device=src.device)
    segment_ids = torch.searchsorted(indptr[1:], indices, right=True)
    segment_ids = segment_ids.clamp(0, num_segments - 1)
    
    if reduce == "sum":
        out = torch.zeros(num_segments, *src.shape[1:], dtype=src.dtype, device=src.device)
        out.scatter_add_(0, segment_ids.view(-1, *([1] * (src.ndim - 1))).expand_as(src), src)
    elif reduce == "mean":
        out = torch.zeros(num_segments, *src.shape[1:], dtype=src.dtype, device=src.device)
        out.scatter_add_(0, segment_ids.view(-1, *([1] * (src.ndim - 1))).expand_as(src), src)
        counts = (indptr[1:] - indptr[:-1]).float().view(-1, *([1] * (src.ndim - 1)))
        out = out / counts.clamp(min=1)
    elif reduce == "max":
        out = torch.full((num_segments, *src.shape[1:]), float('-inf'), dtype=src.dtype, device=src.device)
        out.scatter_reduce_(0, segment_ids.view(-1, *([1] * (src.ndim - 1))).expand_as(src), src, reduce="amax", include_self=True)
        out = torch.where(out == float('-inf'), torch.zeros_like(out), out)
    elif reduce == "min":
        out = torch.full((num_segments, *src.shape[1:]), float('inf'), dtype=src.dtype, device=src.device)
        out.scatter_reduce_(0, segment_ids.view(-1, *([1] * (src.ndim - 1))).expand_as(src), src, reduce="amin", include_self=True)
        out = torch.where(out == float('inf'), torch.zeros_like(out), out)
    else:
        raise ValueError(f"Unknown reduce type: {reduce}")
    
    return out


# =============================================================================
# Normal Estimation Utilities
# =============================================================================

def estimate_normals_open3d(
    coords: np.ndarray,
    radius: float = 0.03,
    max_nn: int = 30,
    camera_location: np.ndarray = None,
) -> np.ndarray:
    """
    Estimate surface normals using Open3D.
    
    Args:
        coords: (N, 3) point coordinates
        radius: Search radius for normal estimation
        max_nn: Maximum neighbors to consider
        camera_location: Optional camera location for normal orientation
    
    Returns:
        normals: (N, 3) estimated normals
    """
    if not HAS_OPEN3D:
        return estimate_normals_fallback(coords)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    
    # Orient normals towards camera (typically at origin for tabletop setup)
    if camera_location is None:
        camera_location = np.array([0.0, 0.0, 0.0])
    pcd.orient_normals_towards_camera_location(camera_location)
    
    normals = np.asarray(pcd.normals).astype(np.float32)
    return normals


def estimate_normals_fallback(coords: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Fallback normal estimation using PCA on k-nearest neighbors.
    Slower than Open3D but pure numpy/torch.
    
    Args:
        coords: (N, 3) point coordinates
        k: Number of neighbors for PCA
    
    Returns:
        normals: (N, 3) estimated normals
    """
    from scipy.spatial import cKDTree
    
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k)
    
    normals = np.zeros_like(coords)
    for i in range(len(coords)):
        neighbors = coords[indices[i]]
        centered = neighbors - neighbors.mean(axis=0)
        _, _, vh = np.linalg.svd(centered)
        normals[i] = vh[2]  # Smallest singular vector is normal
    
    # Orient towards origin (camera)
    view_dirs = -coords  # Direction from point to origin
    dot = np.sum(normals * view_dirs, axis=1)
    normals[dot < 0] *= -1
    
    return normals.astype(np.float32)


def estimate_normals_batched(
    coords: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
    voxel_size: float = 0.005,
) -> torch.Tensor:
    """
    Estimate normals for batched point cloud.
    
    Args:
        coords: (N, 3) voxel coordinates (integer grid)
        batch_idx: (N,) batch index for each point
        batch_size: Number of samples in batch
        voxel_size: Voxel size to convert grid coords to meters
    
    Returns:
        normals: (N, 3) estimated normals
    """
    device = coords.device
    
    # Convert to numpy for Open3D processing
    coords_meters = (coords.float() * voxel_size).cpu().numpy()
    batch_idx_np = batch_idx.cpu().numpy()
    
    all_normals = np.zeros_like(coords_meters)
    
    for b in range(batch_size):
        mask = batch_idx_np == b
        if mask.sum() == 0:
            continue
        
        batch_coords = coords_meters[mask]
        batch_normals = estimate_normals_open3d(
            batch_coords,
            radius=voxel_size * 6,  # ~6 voxels radius
            max_nn=30,
        )
        all_normals[mask] = batch_normals
    
    return torch.from_numpy(all_normals).float().to(device)


# =============================================================================
# Sonata Model Components (encoder-only PTv3)
# =============================================================================

class RPE(nn.Module):
    """Relative Position Encoding."""
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_enc = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, num_heads),
        )

    def forward(self, coord):
        coord = coord.float()
        table = self.pos_enc(coord).view(-1, self.num_heads)
        return table


class SerializedAttention(PointModule):
    """Serialized attention for point clouds."""
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash and HAS_FLASH_ATTN

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)

        if enable_rpe:
            self.rpe = RPE(patch_size, num_heads)

    def forward(self, point):
        H = self.num_heads
        K = point.serialized_code.shape[0]
        C = self.channels

        pad = point.serialized_pad if hasattr(point, 'serialized_pad') else 0

        order = point.serialized_order[self.order_index]
        inverse = point.serialized_inverse[self.order_index]

        if pad > 0:
            padding_mode = 'circular' if hasattr(point, 'padding_mode') and point.padding_mode == 'circular' else 'zeros'
            if padding_mode == 'zeros':
                pad_feat = torch.zeros(pad, C, device=point.feat.device, dtype=point.feat.dtype)
            else:
                pad_feat = point.feat[:pad]
            point_feat = torch.cat([point.feat, pad_feat], dim=0)
        else:
            point_feat = point.feat

        feat = point_feat[order]
        
        qkv = self.qkv(feat).reshape(-1, 3, H, C // H)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        if self.enable_flash:
            from flash_attn import flash_attn_varlen_qkvpacked_func
            qkv_packed = torch.stack([q, k, v], dim=1)
            feat = flash_attn_varlen_qkvpacked_func(
                qkv_packed.half() if qkv_packed.dtype == torch.float32 else qkv_packed,
                point.serialized_indptr if hasattr(point, 'serialized_indptr') else torch.tensor([0, feat.shape[0]], device=feat.device),
                max_seqlen=feat.shape[0],
                dropout_p=0.0 if not self.training else self.attn_drop.p,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        else:
            # Standard attention (fallback)
            q = q.transpose(0, 1)  # (H, N, d)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            feat = (attn @ v).transpose(0, 1).reshape(-1, C)
            
        feat = feat[inverse]
        if pad > 0:
            feat = feat[:-pad]

        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    """Transformer block."""
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(channels, channels, kernel_size=3, bias=True, indice_key=cpe_indice_key),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    """Downsampling via serialized pooling."""
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        else:
            self.norm = None
        if act_layer is not None:
            self.act = PointSequential(act_layer())
        else:
            self.act = None

    def forward(self, point: Point) -> Point:
        # Store parent for unpooling
        if self.traceable:
            point["pooling_parent"] = point.clone()

        # Pooling
        pooling_depth = self.stride.bit_length() - 1
        n_o, count = torch.unique(
            point.serialized_code[0] >> (pooling_depth * 3),
            sorted=True,
            return_counts=True,
        )
        
        n = point.feat.shape[0]
        indptr = torch.cat([torch.tensor([0], device=count.device), count.cumsum(0)])
        
        order = point.serialized_order[0]
        inverse = point.serialized_inverse[0]
        
        feat = point.feat[order]
        feat = segment_csr(feat, indptr, reduce=self.reduce)
        
        # Update point attributes
        point.feat = self.proj(feat)
        
        # Update coordinates
        point.coord = segment_csr(point.coord[order].float(), indptr, reduce="mean")
        point.grid_coord = torch.div(point.grid_coord[order], self.stride, rounding_mode="trunc")
        point.grid_coord = segment_csr(point.grid_coord.float(), indptr, reduce="mean").int()
        
        # Update batch info
        point.batch = segment_csr(point.batch[order].float().unsqueeze(-1), indptr, reduce="min").squeeze(-1).long()
        point.offset = batch2offset(point.batch)
        
        # Store inverse mapping for unpooling
        if self.traceable:
            point["pooling_inverse"] = torch.searchsorted(indptr[1:], inverse.long())
        
        # Re-serialize
        point.serialization(order=point.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        
        return point


class Embedding(PointModule):
    """Point cloud embedding layer using Linear projection (matches Sonata checkpoint)."""
    def __init__(self, in_channels, embed_channels, norm_layer=None, act_layer=None):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # Use Linear layer to match Sonata checkpoint structure
        self.stem = nn.Sequential()
        self.stem.add_module("linear", nn.Linear(in_channels, embed_channels))
        if norm_layer is not None:
            self.stem.add_module("norm", norm_layer(embed_channels))
        if act_layer is not None:
            self.stem.add_module("act", act_layer())

    def forward(self, point: Point) -> Point:
        # Apply linear projection to point features
        point.feat = self.stem(point.feat)
        # Update sparse_conv_feat with new features (required for CPE convolutions)
        if "sparse_conv_feat" in point.keys():
            point.sparse_conv_feat = spconv.SparseConvTensor(
                features=point.feat,
                indices=point.sparse_conv_feat.indices,
                spatial_shape=point.sparse_conv_feat.spatial_shape,
                batch_size=point.sparse_conv_feat.batch_size,
            )
        return point


# =============================================================================
# Sonata Encoder (Main Model)
# =============================================================================

class SonataEncoder(PointModule):
    """
    Sonata: Self-supervised pretrained PTv3 encoder.
    
    Encoder-only architecture for feature extraction.
    After encoding, features must be unpooled back to original resolution.
    """
    
    def __init__(
        self,
        in_channels=9,  # coord (3) + color (3) + normal (3)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),  # matches Sonata checkpoint
        enc_channels=(48, 96, 192, 384, 512),  # matches Sonata pretrained weights
        enc_num_head=(3, 6, 12, 24, 32),  # matches Sonata checkpoint
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else list(order)
        self.shuffle_orders = shuffle_orders
        self.enc_channels = enc_channels

        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # Encoder
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]):sum(enc_depths[:s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        traceable=True,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

    def forward(self, point: Point) -> Point:
        point = self.embedding(point)
        point = self.enc(point)
        return point


# =============================================================================
# Sonata Backbone Wrapper (spconv interface)
# =============================================================================

class SonataBackbone(nn.Module):
    """
    Sonata backbone wrapper for GraspNet pipeline.
    
    Provides spconv.SparseConvTensor interface with automatic:
    - Normal estimation using Open3D
    - Input format conversion (spconv -> Point)
    - Feature unpooling to original resolution
    - Output projection to match expected feature dimension
    
    Args:
        in_channels: Input feature channels from dataset (3=RGB or 6=XYZ+RGB)
        out_channels: Output feature dimension (default: 512 for GraspNet)
        checkpoint_path: Path to Sonata pretrained weights
        enable_flash: Enable flash attention if available
        voxel_size: Voxel size used in dataset (for coordinate scaling)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 512,
        checkpoint_path: Optional[str] = None,
        enable_flash: bool = False,
        voxel_size: float = 0.005,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.voxel_size = voxel_size
        self.in_channels = in_channels
        
        # Sonata config (matching pretrained weights from checkpoint)
        # Store enc_channels for multi-scale fusion
        self.enc_channels = (48, 96, 192, 384, 512)  # matches Sonata pretrained weights
        
        self.encoder = SonataEncoder(
            in_channels=9,  # Always 9: coord + color + normal
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(3, 3, 3, 12, 3),  # From checkpoint config
            enc_channels=self.enc_channels,
            enc_num_head=(3, 6, 12, 24, 32),  # From checkpoint config
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            mlp_ratio=4,
            qkv_bias=True,
            drop_path=0.3,
            enable_flash=enable_flash and HAS_FLASH_ATTN,
        )
        
        # Multi-scale fusion: concatenate all encoder stage outputs
        # After unpooling with concatenation: 512 + 384 + 192 + 96 + 48 = 1232 channels
        # But we only unpool 4 times (4 strides), so only 4 levels get concatenated
        # 512 + 384 + 192 + 96 = 1184 channels (enc4 -> enc0 after 4 unpools)
        fused_channels = sum(self.enc_channels)  # 48+96+192+384+512 = 1232
        self.fusion_proj = nn.Sequential(
            nn.Linear(fused_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        
        # Simple output projection if fusion not used
        self.output_proj = nn.Identity()
        
        # Load pretrained weights
        if checkpoint_path is None:
            checkpoint_path = os.path.join(os.path.dirname(__file__), "sonata", "sonata.pth")
        
        if os.path.exists(checkpoint_path):
            self._load_pretrained(checkpoint_path)
        else:
            print(f"Warning: Sonata checkpoint not found at {checkpoint_path}")
    
    def _load_pretrained(self, checkpoint_path: str):
        """Load Sonata pretrained weights."""
        print(f"Loading Sonata pretrained weights from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Transform keys - Sonata may have different prefixes
        encoder_state_dict = {}
        for key, value in state_dict.items():
            # Strip common prefixes
            new_key = key
            for prefix in ['module.', 'backbone.', 'encoder.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            
            # Map to our encoder
            encoder_state_dict[new_key] = value
        
        # Load what matches
        model_state_dict = self.encoder.state_dict()
        matched_keys = []
        missing_keys = []
        
        for key in model_state_dict.keys():
            if key in encoder_state_dict:
                if model_state_dict[key].shape == encoder_state_dict[key].shape:
                    matched_keys.append(key)
                else:
                    print(f"Shape mismatch for {key}: model={model_state_dict[key].shape}, ckpt={encoder_state_dict[key].shape}")
                    missing_keys.append(key)
            else:
                missing_keys.append(key)
        
        filtered_state_dict = {k: encoder_state_dict[k] for k in matched_keys}
        self.encoder.load_state_dict(filtered_state_dict, strict=False)
        
        print(f"Loaded {len(matched_keys)}/{len(model_state_dict)} parameters from Sonata checkpoint")
        if missing_keys:
            print(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing: {missing_keys}")
    
    def _unpool_features(self, point: Point) -> Point:
        """
        Unpool features from encoded resolution back to original resolution.
        
        Uses multi-scale feature fusion: concatenate features from each level
        during unpooling to preserve hierarchical information.
        """
        # Multi-scale fusion - concatenate features during unpooling
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            
            # Upsample child features to parent resolution
            upsampled_child = point.feat[inverse]
            
            # Concatenate with parent's features (skip connection)
            parent.feat = torch.cat([parent.feat, upsampled_child], dim=-1)
            point = parent
        
        return point
    
    def forward(self, sparse_input: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """
        Forward pass.
        
        Args:
            sparse_input: spconv.SparseConvTensor with:
                - features: (N, 3) RGB features (normalized 0-1)
                - indices: (N, 4) voxel coordinates [batch, x, y, z]
        
        Returns:
            spconv.SparseConvTensor with (N, out_channels) features
        """
        device = sparse_input.features.device
        batch_size = sparse_input.batch_size
        
        # Extract from sparse tensor
        feats = sparse_input.features  # (N, C) - typically RGB
        indices = sparse_input.indices  # (N, 4) [batch, x, y, z]
        spatial_shape = sparse_input.spatial_shape
        
        N = feats.shape[0]
        
        # Convert grid coordinates to physical coordinates (meters)
        grid_coord = indices[:, 1:4].int()
        coord = grid_coord.float() * self.voxel_size
        batch_idx = indices[:, 0].long()
        
        # Prepare color features (ensure 3 channels)
        if feats.shape[1] == 3:
            color = feats  # RGB
        elif feats.shape[1] == 6:
            color = feats[:, 3:6]  # Assume [xyz, rgb] format, take rgb
        else:
            color = feats[:, :3]  # Take first 3 channels
        
        # Estimate normals (the key addition for Sonata)
        normals = estimate_normals_batched(
            grid_coord,
            batch_idx,
            batch_size,
            voxel_size=self.voxel_size,
        )
        
        # Normalize coordinates to [-1, 1] range for each batch
        coord_normalized = coord.clone()
        for b in range(batch_size):
            mask = batch_idx == b
            if mask.sum() > 0:
                batch_coord = coord[mask]
                coord_min = batch_coord.min(dim=0).values
                coord_max = batch_coord.max(dim=0).values
                coord_range = (coord_max - coord_min).clamp(min=1e-6)
                coord_normalized[mask] = (batch_coord - coord_min) / coord_range * 2 - 1
        
        # Combine features: coord (normalized) + color + normal = 9 channels
        feat = torch.cat([coord_normalized, color, normals], dim=1)  # (N, 9)
        
        # Compute batch offset
        batch_counts = torch.bincount(batch_idx, minlength=batch_size)
        offset = torch.cumsum(batch_counts, dim=0).to(device)
        
        # Create Point structure
        data_dict = dict(
            feat=feat,
            coord=coord,
            grid_coord=grid_coord,
            batch=batch_idx,
            offset=offset,
            grid_size=self.voxel_size,
        )
        
        point = Point(data_dict)
        point.serialization(order=self.encoder.order, shuffle_orders=self.encoder.shuffle_orders)
        point.sparsify()
        
        # Run encoder
        point = self.encoder(point)
        
        # Unpool back to original resolution with multi-scale fusion
        point = self._unpool_features(point)
        
        # Project fused multi-scale features to output dimension
        # point.feat now has 1232 channels (48+96+192+384+512) from concatenation
        out_feat = self.fusion_proj(point.feat)
        
        # Return as sparse tensor with original structure
        out = spconv.SparseConvTensor(
            out_feat,
            indices,
            spatial_shape,
            batch_size,
        )
        
        return out


# =============================================================================
# Factory Function
# =============================================================================

def create_sonata_backbone(
    out_channels: int = 512,
    checkpoint_path: Optional[str] = None,
    enable_flash: bool = False,
    voxel_size: float = 0.005,
    freeze_backbone: bool = False,
) -> SonataBackbone:
    """
    Create Sonata backbone for grasp detection.
    
    Args:
        out_channels: Output feature dimension
        checkpoint_path: Path to pretrained weights (None = use default location)
        enable_flash: Enable flash attention
        voxel_size: Voxel size from dataset
        freeze_backbone: Freeze encoder weights (only train output projection)
    
    Returns:
        SonataBackbone instance with loaded pretrained weights
    """
    model = SonataBackbone(
        in_channels=3,  # Dataset provides RGB
        out_channels=out_channels,
        checkpoint_path=checkpoint_path,
        enable_flash=enable_flash,
        voxel_size=voxel_size,
    )
    
    if freeze_backbone:
        for param in model.encoder.parameters():
            param.requires_grad = False
        # Keep fusion projection trainable
        for param in model.fusion_proj.parameters():
            param.requires_grad = True
    
    return model
