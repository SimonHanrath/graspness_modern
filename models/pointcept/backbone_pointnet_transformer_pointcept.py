"""
Point Transformer V3 Backbone (Encoder-Only)

Adapted from Pointcept implementation by Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Modified for GraspNet pipeline with spconv.SparseConvTensor I/O.

This is a simplified encoder-only version for feature extraction.
Decoder removed since we only need per-point features for grasp detection.
"""

#TODO: verify code, understand "export" structure, figure out hyerparams, also we still have the JITs from spconv and I cant fix that easily maybe we can downgrade spconv or change to a newer version or something

from functools import partial
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv

try:
    import flash_attn
except ImportError:
    flash_attn = None

from .misc import offset2bincount
from .structure import Point
from .modules import PointModule, PointSequential


# =============================================================================
# Inline replacements for external dependencies
# =============================================================================

class DropPath(nn.Module):
    """
    Stochastic Depth (drop path) regularization.
    Replaces timm.layers.DropPath - identical functionality.
    
    Reference: https://arxiv.org/abs/1603.09382
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with any tensor dim - create shape (batch, 1, 1, ...)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    """
    Segment reduction using CSR (Compressed Sparse Row) index pointers.
    Replaces torch_scatter.segment_csr - pure PyTorch implementation.
    
    Args:
        src: Source tensor of shape (N, ...) with values to reduce
        indptr: Index pointer tensor of shape (num_segments + 1,)
                indptr[i] to indptr[i+1] defines segment i
        reduce: Reduction operation - "sum", "mean", "max", "min"
    
    Returns:
        Reduced tensor of shape (num_segments, ...)
    """
    num_segments = indptr.shape[0] - 1
    
    if num_segments == 0:
        # Edge case: no segments
        return torch.zeros((0, *src.shape[1:]), dtype=src.dtype, device=src.device)
    
    if src.shape[0] == 0:
        # Edge case: no source elements
        return torch.zeros((num_segments, *src.shape[1:]), dtype=src.dtype, device=src.device)
    
    # Create segment indices for each element using vectorized approach
    # This replaces the slow Python loop
    segment_ids = torch.zeros(src.shape[0], dtype=torch.long, device=src.device)
    
    # Vectorized segment ID assignment
    # Create a range tensor and use searchsorted to find segment for each index
    indices = torch.arange(src.shape[0], device=src.device)
    segment_ids = torch.searchsorted(indptr[1:], indices, right=True)
    
    # Clamp to valid range (handles edge cases)
    segment_ids = segment_ids.clamp(0, num_segments - 1)
    
    # Handle different reduction types
    if reduce == "sum":
        out = torch.zeros(num_segments, *src.shape[1:], dtype=src.dtype, device=src.device)
        out.scatter_add_(0, segment_ids.view(-1, *([1] * (src.ndim - 1))).expand_as(src), src)
    elif reduce == "mean":
        out = torch.zeros(num_segments, *src.shape[1:], dtype=src.dtype, device=src.device)
        out.scatter_add_(0, segment_ids.view(-1, *([1] * (src.ndim - 1))).expand_as(src), src)
        counts = (indptr[1:] - indptr[:-1]).float().view(-1, *([1] * (src.ndim - 1)))
        out = out / counts.clamp(min=1)
    elif reduce == "max":
        out = torch.zeros(num_segments, *src.shape[1:], dtype=src.dtype, device=src.device)
        out.scatter_reduce_(0, segment_ids.view(-1, *([1] * (src.ndim - 1))).expand_as(src), src, reduce="amax", include_self=False)
    elif reduce == "min":
        out = torch.zeros(num_segments, *src.shape[1:], dtype=src.dtype, device=src.device)
        out.scatter_reduce_(0, segment_ids.view(-1, *([1] * (src.ndim - 1))).expand_as(src), src, reduce="amin", include_self=False)
    else:
        raise ValueError(f"Unknown reduce type: {reduce}")
    
    return out


# =============================================================================
# Model Components
# =============================================================================

class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
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
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.to(torch.bfloat16).reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
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
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

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
    """Downsampling via serialized pooling using space-filling curves."""
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

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = None
        self.act = None
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information (using plain dict instead of addict.Dict)
        point_dict = dict(
            feat=segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class Embedding(PointModule):
    """Initial feature embedding via sparse convolution."""
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


class PointTransformerV3Encoder(PointModule):
    """
    Point Transformer V3 - Encoder Only
    
    Simplified for grasp detection: encoder-only feature extraction.
    Input/Output: spconv.SparseConvTensor for compatibility with GraspNet pipeline.
    
    Args:
        in_channels: Input feature dimension (default: 3 for RGB or XYZ)
        out_channels: Output feature dimension (default: 512)
        order: Serialization order(s) for attention
        stride: Downsampling strides between encoder stages
        enc_depths: Number of transformer blocks per encoder stage
        enc_channels: Feature channels per encoder stage
        enc_num_head: Number of attention heads per encoder stage
        enc_patch_size: Patch size for serialized attention per stage
        mlp_ratio: MLP hidden dim ratio
        qkv_bias: Use bias in QKV projection
        qk_scale: Override default QK scale
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
        drop_path: Stochastic depth rate
        pre_norm: Use pre-normalization (vs post-norm)
        shuffle_orders: Shuffle serialization orders during training
        enable_rpe: Enable relative position encoding
        enable_flash: Enable flash attention (requires flash_attn)
        upcast_attention: Upcast attention to float32
        upcast_softmax: Upcast softmax to float32
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=512,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
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
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders
        self.out_channels = out_channels

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)

        # Normalization layers (simplified - no PDNorm)
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
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
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

        # Output projection to match expected output channels
        final_enc_channels = enc_channels[-1]
        if final_enc_channels != out_channels:
            self.output_proj = nn.Linear(final_enc_channels, out_channels)
        else:
            self.output_proj = nn.Identity()

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """
        Forward pass compatible with spconv SparseConvTensor.
        
        Args:
            x: spconv.SparseConvTensor with:
                - features: (N, in_channels) voxel features
                - indices: (N, 4) voxel coordinates [batch, z, y, x]
                - spatial_shape: (D, H, W)
                - batch_size: B
        
        Returns:
            spconv.SparseConvTensor with:
                - features: (N, out_channels) per-point features
                - Same indices, spatial_shape, batch_size as input
        """
        # Extract data from sparse tensor
        feat = x.features  # (N, in_channels)
        indices = x.indices  # (N, 4) [batch, z, y, x]
        spatial_shape = x.spatial_shape
        batch_size = x.batch_size
        
        device = feat.device
        N = feat.shape[0]
        
        # Convert to Point format expected by PTv3
        # coord: continuous coordinates (use grid coords scaled)
        # grid_coord: integer grid coordinates
        coord = indices[:, 1:].float()  # (N, 3) [z, y, x]
        grid_coord = indices[:, 1:].int()  # (N, 3)
        batch = indices[:, 0].long()  # (N,)
        
        # Compute offset (cumulative count per batch)
        # offset[i] = total points in batches 0..i
        batch_counts = torch.bincount(batch, minlength=batch_size)
        offset = torch.cumsum(batch_counts, dim=0).to(device)
        
        # Build data dict for Point
        data_dict = dict(
            feat=feat,
            coord=coord,
            grid_coord=grid_coord,
            batch=batch,
            offset=offset,
            grid_size=1.0,  # Already voxelized
        )
        
        # Create Point and process
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        
        # Run encoder
        point = self.embedding(point)
        point = self.enc(point)
        
        # NOTE: After pooling, we have fewer points than input.
        # For grasp detection, we need features at original resolution.
        # Two options:
        # 1. Use traceable=True and upsample back (need to implement)
        # 2. Use the final pooled features (for global reasoning)
        # 
        # For now, we return the pooled features.
        # If you need original resolution, we'd need to add upsampling.
        
        # Get output features
        out_feat = self.output_proj(point.feat)
        
        # Return as sparse tensor
        # Note: indices have changed due to pooling
        out_indices = torch.cat([
            point.batch.unsqueeze(1),
            point.grid_coord
        ], dim=1).int()
        
        # Spatial shape may need adjustment based on pooling
        # For now, use original (sparse tensor handles this)
        out = spconv.SparseConvTensor(
            out_feat,
            out_indices,
            spatial_shape,
            batch_size
        )
        
        return out


class PointTransformerV3EncoderFullRes(PointModule):
    """
    Point Transformer V3 - Encoder with Full Resolution Output
    
    Uses encoder-decoder to get features at original point resolution.
    This is needed for dense prediction tasks like grasp detection.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension (default: 512)
        ... (same as PointTransformerV3Encoder)
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=512,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
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
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders
        self.out_channels = out_channels

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.num_stages == len(dec_depths) + 1
        assert self.num_stages == len(dec_channels) + 1
        assert self.num_stages == len(dec_num_head) + 1
        assert self.num_stages == len(dec_patch_size) + 1

        # Normalization layers
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
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        traceable=True,  # Enable for unpooling
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

        # Decoder (for upsampling back to original resolution)
        dec_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
        ]
        self.dec = PointSequential()
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        for s in reversed(range(self.num_stages - 1)):
            dec_drop_path_ = dec_drop_path[
                sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
            ]
            dec_drop_path_.reverse()
            dec = PointSequential()
            dec.add(
                SerializedUnpooling(
                    in_channels=dec_channels[s + 1],
                    skip_channels=enc_channels[s],
                    out_channels=dec_channels[s],
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                ),
                name="up",
            )
            for i in range(dec_depths[s]):
                dec.add(
                    Block(
                        channels=dec_channels[s],
                        num_heads=dec_num_head[s],
                        patch_size=dec_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=dec_drop_path_[i],
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
            self.dec.add(module=dec, name=f"dec{s}")

        # Output projection
        final_dec_channels = dec_channels[0]
        self.output_proj = nn.Linear(final_dec_channels, out_channels)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """
        Forward pass with full resolution output.
        
        Args:
            x: spconv.SparseConvTensor with:
                - features: (N, in_channels) voxel features
                - indices: (N, 4) voxel coordinates [batch, z, y, x]
                - spatial_shape: (D, H, W)
                - batch_size: B
        
        Returns:
            spconv.SparseConvTensor with:
                - features: (N, out_channels) per-point features at original resolution
                - Same indices, spatial_shape, batch_size as input
        """
        # Store original indices for output
        original_indices = x.indices
        spatial_shape = x.spatial_shape
        batch_size = x.batch_size
        
        # Extract data from sparse tensor
        feat = x.features  # (N, in_channels)
        indices = x.indices  # (N, 4) [batch, z, y, x]
        
        device = feat.device
        N = feat.shape[0]
        
        # Convert to Point format
        coord = indices[:, 1:].float()  # (N, 3)
        grid_coord = indices[:, 1:].int()  # (N, 3)
        batch = indices[:, 0].long()  # (N,)
        
        # Compute offset
        batch_counts = torch.bincount(batch, minlength=batch_size)
        offset = torch.cumsum(batch_counts, dim=0).to(device)
        
        # Build data dict for Point
        data_dict = dict(
            feat=feat,
            coord=coord,
            grid_coord=grid_coord,
            batch=batch,
            offset=offset,
            grid_size=1.0,
        )
        
        # Create Point and process
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        
        # Run encoder-decoder
        point = self.embedding(point)
        point = self.enc(point)
        point = self.dec(point)
        
        # Get output features
        out_feat = self.output_proj(point.feat)
        
        # Return as sparse tensor with original structure
        out = spconv.SparseConvTensor(
            out_feat,
            original_indices,
            spatial_shape,
            batch_size
        )
        
        return out


# We need SerializedUnpooling for the full-res version
class SerializedUnpooling(PointModule):
    """Upsampling via unpooling using stored parent/inverse info."""
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]
        return parent


# =============================================================================
# Pretrained Weight Loading
# =============================================================================

def load_pointcept_pretrained_backbone(
    model: PointTransformerV3EncoderFullRes,
    checkpoint_path: str,
    strict: bool = False,
) -> PointTransformerV3EncoderFullRes:
    """
    Load pretrained Pointcept backbone weights into the model.
    
    The pretrained checkpoint from Pointcept has keys like:
        'module.backbone.embedding.stem.conv.weight'
        'module.backbone.enc.enc0.block0.attn.qkv.weight'
        ...
    
    We need to strip 'module.backbone.' prefix to match our model's state dict.
    
    Args:
        model: PointTransformerV3EncoderFullRes instance to load weights into
        checkpoint_path: Path to the .pth checkpoint file
        strict: If True, raise error on missing/unexpected keys. If False, only load matching keys.
    
    Returns:
        The model with loaded weights
    """
    import os
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get state dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Transform keys: strip 'module.backbone.' prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        # Skip seg_head keys - we don't need them for backbone
        if 'seg_head' in key:
            continue
        
        # Strip prefix
        if key.startswith('module.backbone.'):
            new_key = key[len('module.backbone.'):]
        elif key.startswith('backbone.'):
            new_key = key[len('backbone.'):]
        elif key.startswith('module.'):
            new_key = key[len('module.'):]
        else:
            new_key = key
        
        new_state_dict[new_key] = value
    
    # Get model's current state dict
    model_state_dict = model.state_dict()
    
    # Filter to only matching keys
    matched_keys = []
    missing_keys = []
    unexpected_keys = []
    
    for key in model_state_dict.keys():
        if key in new_state_dict:
            if model_state_dict[key].shape == new_state_dict[key].shape:
                matched_keys.append(key)
            else:
                print(f"Shape mismatch for {key}: model={model_state_dict[key].shape}, checkpoint={new_state_dict[key].shape}")
                missing_keys.append(key)
        else:
            missing_keys.append(key)
    
    for key in new_state_dict.keys():
        if key not in model_state_dict:
            unexpected_keys.append(key)
    
    # Load matching weights
    filtered_state_dict = {k: new_state_dict[k] for k in matched_keys}
    model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"Loaded {len(matched_keys)} / {len(model_state_dict)} parameters from pretrained checkpoint")
    if missing_keys:
        print(f"Missing keys ({len(missing_keys)}): {missing_keys[:10]}..." if len(missing_keys) > 10 else f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}..." if len(unexpected_keys) > 10 else f"Unexpected keys: {unexpected_keys}")
    
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(f"Strict loading failed: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys")
    
    return model


def create_ptv3_backbone_from_pretrained(
    checkpoint_path: str,
    out_channels: int = 512,
    freeze_backbone: bool = False,
    **override_kwargs,
) -> PointTransformerV3EncoderFullRes:
    """
    Create a PTv3 backbone with architecture matching the pretrained Pointcept checkpoint.
    
    The pretrained model (ScanNet semantic segmentation) has:
    - in_channels: 6 (color + normal or XYZ + color)
    - enc_depths: (2, 2, 2, 6, 2)
    - enc_channels: (32, 64, 128, 256, 512)
    - enc_num_head: (2, 4, 8, 16, 32)
    - dec_depths: (2, 2, 2, 2)
    - dec_channels: (64, 64, 128, 256)
    - dec_num_head: (4, 4, 8, 16)
    
    Args:
        checkpoint_path: Path to the pretrained .pth file
        out_channels: Output feature dimension (adds projection layer if different from dec output)
        freeze_backbone: If True, freeze all backbone parameters
        **override_kwargs: Override any default architecture parameters
    
    Returns:
        PointTransformerV3EncoderFullRes with pretrained weights loaded
    """
    # Default config matching pretrained checkpoint
    default_config = dict(
        in_channels=6,
        out_channels=out_channels,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
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
    )
    
    # Override with user-specified kwargs
    default_config.update(override_kwargs)
    
    # Create model
    model = PointTransformerV3EncoderFullRes(**default_config)
    
    # Load pretrained weights
    load_pointcept_pretrained_backbone(model, checkpoint_path, strict=False)
    
    # Optionally freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze output projection if it exists and is newly added
        if hasattr(model, 'output_proj') and isinstance(model.output_proj, nn.Linear):
            for param in model.output_proj.parameters():
                param.requires_grad = True
    
    return model


def create_ptv3_backbone_grasp(
    checkpoint_path: str = None,
    in_channels: int = 3,
    out_channels: int = 512,
    use_pretrained: bool = True,
    enable_flash: bool = False,
    **kwargs,
) -> PointTransformerV3EncoderFullRes:
    """
    Create a PTv3 backbone configured for grasp detection.
    
    This handles the input channel mismatch: pretrained model expects 6 channels
    (RGB + normal or XYZ + RGB), but grasp detection may use 3 channels (XYZ or RGB).
    
    When in_channels != 6 and using pretrained weights:
    - Creates model with 6 input channels to load pretrained weights
    - Replaces the embedding layer with one that matches in_channels
    - The new embedding layer is randomly initialized
    
    Args:
        checkpoint_path: Path to pretrained checkpoint (None to use default location)
        in_channels: Input feature channels (3 for XYZ/RGB, 6 for XYZ+RGB, etc.)
        out_channels: Output feature dimension
        use_pretrained: Whether to load pretrained weights
        enable_flash: Enable flash attention (requires flash_attn package)
        **kwargs: Additional architecture overrides
    
    Returns:
        PointTransformerV3EncoderFullRes configured for grasp detection
    """
    import os
    
    # Default checkpoint path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), 'model_best.pth'
        )
    
    if use_pretrained and os.path.exists(checkpoint_path):
        # Load pretrained model
        model = create_ptv3_backbone_from_pretrained(
            checkpoint_path=checkpoint_path,
            out_channels=out_channels,
            enable_flash=enable_flash,
            **kwargs,
        )
        
        # Handle input channel mismatch
        if in_channels != 6:
            print(f"Replacing embedding layer: {6} -> {in_channels} input channels")
            # Save the pretrained embedding parameters we can reuse
            # (only if we're reducing channels, we can copy a subset)
            
            # Create new embedding with correct input channels
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
            act_layer = nn.GELU
            embed_channels = model.embedding.embed_channels
            
            new_embedding = Embedding(
                in_channels=in_channels,
                embed_channels=embed_channels,
                norm_layer=bn_layer,
                act_layer=act_layer,
            )
            
            # Try to copy what we can from pretrained
            if in_channels <= 6:
                with torch.no_grad():
                    # Copy first in_channels of the conv weights
                    old_weight = model.embedding.stem.conv.weight  # [out, kD, kH, kW, in]
                    new_embedding.stem.conv.weight.copy_(old_weight[:, :, :, :, :in_channels])
                    # Copy norm parameters
                    if hasattr(model.embedding.stem, 'norm'):
                        new_embedding.stem.norm.weight.copy_(model.embedding.stem.norm.weight)
                        new_embedding.stem.norm.bias.copy_(model.embedding.stem.norm.bias)
                        new_embedding.stem.norm.running_mean.copy_(model.embedding.stem.norm.running_mean)
                        new_embedding.stem.norm.running_var.copy_(model.embedding.stem.norm.running_var)
            
            model.embedding = new_embedding
    else:
        # Create model from scratch
        model = PointTransformerV3EncoderFullRes(
            in_channels=in_channels,
            out_channels=out_channels,
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
            enable_flash=enable_flash,
            **kwargs,
        )
        
        if use_pretrained:
            print(f"Warning: Pretrained checkpoint not found at {checkpoint_path}")
    
    return model