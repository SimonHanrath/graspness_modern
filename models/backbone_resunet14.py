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

        self.final = spconv.SubMConv3d(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            indice_key="subm_p1"
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, debug_voxel_counts=False, debug_feature_stats=False):
        out = self.conv0p1s1(x)
        out = out.replace_feature(self.bn0(out.features))
        out_p1 = out.replace_feature(self.relu(out.features))
        if debug_voxel_counts:
            print(f"[SPCONV] out_p1 (stride 1): {out_p1.features.shape[0]} active voxels, spatial_shape={out_p1.spatial_shape}")

        out = self.conv1p1s2(out_p1)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))
        out_b1p2 = self.block1(out)
        if debug_voxel_counts:
            print(f"[SPCONV] out_b1p2 (stride 2): {out_b1p2.features.shape[0]} active voxels, spatial_shape={out_b1p2.spatial_shape}")

        out = self.conv2p2s2(out_b1p2)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features))
        out_b2p4 = self.block2(out)
        if debug_voxel_counts:
            print(f"[SPCONV] out_b2p4 (stride 4): {out_b2p4.features.shape[0]} active voxels, spatial_shape={out_b2p4.spatial_shape}")

        out = self.conv3p4s2(out_b2p4)
        out = out.replace_feature(self.bn3(out.features))
        out = out.replace_feature(self.relu(out.features))
        out_b3p8 = self.block3(out)
        if debug_voxel_counts:
            print(f"[SPCONV] out_b3p8 (stride 8): {out_b3p8.features.shape[0]} active voxels, spatial_shape={out_b3p8.spatial_shape}")

        out = self.conv4p8s2(out_b3p8)
        out = out.replace_feature(self.bn4(out.features))
        out = out.replace_feature(self.relu(out.features))
        out = self.block4(out)
        if debug_voxel_counts:
            print(f"[SPCONV] bottleneck (stride 16): {out.features.shape[0]} active voxels, spatial_shape={out.spatial_shape}")

        out = self.convtr4p16s2(out)
        out = out.replace_feature(self.bntr4(out.features))
        out = out.replace_feature(self.relu(out.features))
        if debug_voxel_counts:
            print(f"[SPCONV] upsample_p8 (after convtr4): {out.features.shape[0]} active voxels, spatial_shape={out.spatial_shape}")

        out = sparse_cat(out, out_b3p8)
        out = self.block5(out)

        out = self.convtr5p8s2(out)
        out = out.replace_feature(self.bntr5(out.features))
        out = out.replace_feature(self.relu(out.features))
        if debug_voxel_counts:
            print(f"[SPCONV] upsample_p4 (after convtr5): {out.features.shape[0]} active voxels, spatial_shape={out.spatial_shape}")

        out = sparse_cat(out, out_b2p4)
        out = self.block6(out)

        out = self.convtr6p4s2(out)
        out = out.replace_feature(self.bntr6(out.features))
        out = out.replace_feature(self.relu(out.features))
        if debug_voxel_counts:
            print(f"[SPCONV] upsample_p2 (after convtr6): {out.features.shape[0]} active voxels, spatial_shape={out.spatial_shape}")

        out = sparse_cat(out, out_b1p2)
        out = self.block7(out)

        out = self.convtr7p2s2(out)
        out = out.replace_feature(self.bntr7(out.features))
        out = out.replace_feature(self.relu(out.features))
        if debug_voxel_counts:
            print(f"[SPCONV] upsample_p1 (after convtr7): {out.features.shape[0]} active voxels, spatial_shape={out.spatial_shape}")

        out = sparse_cat(out, out_p1)
        out = self.block8(out)
        out = self.final(out)
        if debug_voxel_counts:
            print(f"[SPCONV] final output: {out.features.shape[0]} active voxels, spatial_shape={out.spatial_shape}")
        #out = out.replace_feature(self.relu(out.features))
        
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
