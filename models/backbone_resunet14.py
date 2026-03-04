import torch
import torch.nn as nn
import spconv.pytorch as spconv
from models.resnet import ResNetBase


def sparse_cat(a: spconv.SparseConvTensor, b: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
    if not torch.equal(a.indices, b.indices):
        raise ValueError("sparse_cat: indices differ; ensure matching indice_key / inverse conv alignment.")
    if a.spatial_shape != b.spatial_shape or a.batch_size != b.batch_size:
        raise ValueError("sparse_cat: spatial_shape or batch_size mismatch.")

    return a.replace_feature(torch.cat([a.features, b.features], dim=1))


def gather_features_at_indices(
    expanded: spconv.SparseConvTensor,
    original_indices: torch.Tensor
) -> spconv.SparseConvTensor:
    """
    Extract features from an expanded SparseConvTensor at the original voxel locations.
    
    After SparseConv3d (stride=1), the output may have more voxels than the input.
    This function gathers features only at the original voxel positions so that
    the output can be properly mapped back to the original point cloud.
    
    Args:
        expanded: SparseConvTensor after expansion (M' voxels, M' >= M)
        original_indices: (M, 4) tensor of [batch, x, y, z] indices from before expansion
    
    Returns:
        SparseConvTensor with M voxels at the original locations
    """
    exp_indices = expanded.indices  # (M', 4)
    spatial_shape = expanded.spatial_shape
    
    # Create unique keys for fast lookup: batch * (X*Y*Z) + x * (Y*Z) + y * Z + z
    # Using int64 to avoid overflow
    X, Y, Z = spatial_shape
    multiplier = torch.tensor(
        [X * Y * Z, Y * Z, Z, 1],
        device=exp_indices.device,
        dtype=torch.int64
    )
    
    exp_keys = (exp_indices.to(torch.int64) * multiplier).sum(dim=1)  # (M',)
    orig_keys = (original_indices.to(torch.int64) * multiplier).sum(dim=1)  # (M,)
    
    # Build a mapping from key -> position in expanded tensor
    # Since expanded includes all original voxels (conv can only add, not remove),
    # every orig_key should exist in exp_keys
    
    # Create lookup table: for each unique key, store its position in exp_indices
    max_key = exp_keys.max().item() + 1
    lookup = torch.full((max_key,), -1, dtype=torch.long, device=exp_indices.device)
    lookup[exp_keys] = torch.arange(len(exp_keys), device=exp_indices.device)
    
    # Gather indices for original positions
    gather_idx = lookup[orig_keys]  # (M,)
    
    # Sanity check: all original voxels should be found
    if (gather_idx < 0).any():
        raise RuntimeError(
            "gather_features_at_indices: some original voxels not found in expanded tensor. "
            "This shouldn't happen with SparseConv3d(stride=1)."
        )
    
    gathered_features = expanded.features[gather_idx]  # (M, C)
    
    # Create new SparseConvTensor with original indices and gathered features
    return spconv.SparseConvTensor(
        features=gathered_features,
        indices=original_indices,
        spatial_shape=spatial_shape,
        batch_size=expanded.batch_size
    )

class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1, indice_key=None):
        super().__init__()

        ConvLayer = spconv.SubMConv3d if stride == 1 else spconv.SparseConv3d

        self.conv1 = ConvLayer(
            inplanes, planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            indice_key=f"{indice_key}_1" if indice_key else None
        )
        self.norm1 = nn.BatchNorm1d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = spconv.SubMConv3d(
            planes, planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
            indice_key=f"{indice_key}_2" if indice_key else None
        )
        self.norm2 = nn.BatchNorm1d(planes, momentum=bn_momentum)

        self.downsample = downsample

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.norm1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.norm2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)

        out = out.replace_feature(self.relu(out.features))
        
        return out

class SPconvUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = spconv.SubMConv3d(
            in_channels, self.inplanes, kernel_size=5, stride=1, padding=2, bias=False,
            indice_key="subm_p1"
        )
        self.bn0 = nn.BatchNorm1d(self.inplanes)

        self.conv1p1s2 = spconv.SparseConv3d(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, bias=False,
            indice_key="enc_p2"
        )
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0], indice_key_prefix="subm_p2")

        self.conv2p2s2 = spconv.SparseConv3d(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, bias=False,
            indice_key="enc_p4"
        )
        self.bn2 = nn.BatchNorm1d(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1], indice_key_prefix="subm_p4")

        self.conv3p4s2 = spconv.SparseConv3d(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, bias=False,
            indice_key="enc_p8"
        )
        self.bn3 = nn.BatchNorm1d(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2], indice_key_prefix="subm_p8")

        self.conv4p8s2 = spconv.SparseConv3d( #TODO: this leads to problems if we do not have enough points per voxel as we downsample to ahrd, so I replace it for now
            self.inplanes, self.inplanes, kernel_size=2, stride=2, bias=False,
            indice_key="enc_p16"
        )

        """self.conv4p8s2 = spconv.SubMConv3d(
            self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1,
            indice_key="subm_p8_bottleneck"
        )"""


        self.bn4 = nn.BatchNorm1d(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3], indice_key_prefix="subm_p16")

        self.convtr4p16s2 = spconv.SparseInverseConv3d(
            self.inplanes, self.PLANES[4], kernel_size=2, bias=False,
            indice_key="enc_p16"
        )
        self.bntr4 = nn.BatchNorm1d(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4], indice_key_prefix="subm_p8")

        self.convtr5p8s2 = spconv.SparseInverseConv3d(
            self.inplanes, self.PLANES[5], kernel_size=2, bias=False,
            indice_key="enc_p8"
        )
        self.bntr5 = nn.BatchNorm1d(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5], indice_key_prefix="subm_p4")

        self.convtr6p4s2 = spconv.SparseInverseConv3d(
            self.inplanes, self.PLANES[6], kernel_size=2, bias=False,
            indice_key="enc_p4"
        )
        self.bntr6 = nn.BatchNorm1d(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6], indice_key_prefix="subm_p2")

        self.convtr7p2s2 = spconv.SparseInverseConv3d(
            self.inplanes, self.PLANES[7], kernel_size=2, bias=False,
            indice_key="enc_p2"
        )
        self.bntr7 = nn.BatchNorm1d(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7], indice_key_prefix="subm_p1")

        # Refinement block: SparseConv3d allows feature propagation to neighbor voxels
        # This mimics ME's MinkowskiConvolution behavior for better feature aggregation
        # Placed after all inverse convs so it doesn't break the encoder-decoder index chain
        refine_channels = self.PLANES[7] * self.BLOCK.expansion
        self.refine = spconv.SparseSequential(
            spconv.SparseConv3d(
                refine_channels,
                refine_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                indice_key="refine_expand"  # New key, structure changes here
            ),
            nn.BatchNorm1d(refine_channels),
            nn.ReLU(inplace=True),
            # SubM conv after expansion to refine without further expansion
            spconv.SubMConv3d(
                refine_channels,
                refine_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                indice_key="refine_subm"
            ),
            nn.BatchNorm1d(refine_channels),
            nn.ReLU(inplace=True),
        )

        self.final = spconv.SubMConv3d(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            indice_key="refine_subm"  # Use expanded indices for final output
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = out.replace_feature(self.bn0(out.features))
        out_p1 = out.replace_feature(self.relu(out.features))

        out = self.conv1p1s2(out_p1)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features))
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = out.replace_feature(self.bn3(out.features))
        out = out.replace_feature(self.relu(out.features))
        out_b3p8 = self.block3(out)

        out = self.conv4p8s2(out_b3p8)
        out = out.replace_feature(self.bn4(out.features))
        out = out.replace_feature(self.relu(out.features))
        out = self.block4(out)

        out = self.convtr4p16s2(out)
        out = out.replace_feature(self.bntr4(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = sparse_cat(out, out_b3p8)
        out = self.block5(out)

        out = self.convtr5p8s2(out)
        out = out.replace_feature(self.bntr5(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = sparse_cat(out, out_b2p4)
        out = self.block6(out)

        out = self.convtr6p4s2(out)
        out = out.replace_feature(self.bntr6(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = sparse_cat(out, out_b1p2)
        out = self.block7(out)

        out = self.convtr7p2s2(out)
        out = out.replace_feature(self.bntr7(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = sparse_cat(out, out_p1)
        out = self.block8(out)
        
        # Save original indices before refinement expansion
        original_indices = out.indices.clone()
        
        # Apply refinement (SparseConv3d may expand voxel set)
        out = self.refine(out)
        
        # Apply final conv while still on expanded indices (consistent with refine's indice_key)
        out = self.final(out)
        
        # Gather features back at original voxel locations AFTER final
        # This ensures output indices match input indices for proper quantize2original mapping
        out = gather_features_at_indices(out, original_indices)
        
        return out


class SPconvUNet14(SPconvUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class SPconvUNet18(SPconvUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class SPconvUNet34(SPconvUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class SPconvUNet14A(SPconvUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class SPconvUNet14B(SPconvUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class SPconvUNet14C(SPconvUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class SPconvUNet14Dori(SPconvUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class SPconvUNet14E(SPconvUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class SPconvUNet14D(SPconvUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 192, 192)


class SPconvUNet18A(SPconvUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class SPconvUNet18B(SPconvUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class SPconvUNet18D(SPconvUNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class SPconvUNet34A(SPconvUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class SPconvUNet34B(SPconvUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class SPconvUNet34C(SPconvUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
