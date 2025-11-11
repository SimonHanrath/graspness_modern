import torch
import torch.nn as nn
import spconv.pytorch as spconv

class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):

        self.inplanes = self.INIT_DIM
        self.conv1 = spconv.SparseSequential(
            spconv.SparseConv3d(
                in_channels, self.inplanes,
                kernel_size=3, stride=2, padding=1, bias=False,
                indice_key="res_enc1"
            ),
            nn.BatchNorm1d(self.inplanes),
            nn.ReLU(inplace=True),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2, indice_key="res_pool1")
        )

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2, indice_key_prefix="res_stage1"
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2, indice_key_prefix="res_stage2"
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2, indice_key_prefix="res_stage3"
        )
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2, indice_key_prefix="res_stage4"
        )

        self.conv5 = spconv.SparseSequential(
            nn.Dropout(p=0.5),
            spconv.SparseConv3d(
                self.inplanes, self.inplanes,
                kernel_size=3, stride=3, bias=False,
                indice_key="res_down5"
            ),
            nn.BatchNorm1d(self.inplanes),
            nn.GELU()
        )

        self.glob_pool = spconv.SparseGlobalMaxPool()
        self.final = nn.Linear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, (spconv.SparseConv3d, spconv.SubMConv3d, spconv.SparseInverseConv3d)):
                # spconv weight shape: [out_channels, kH, kW, kD, in_channels]
                # We need to manually calculate fan_in and fan_out for correct initialization
                weight = m.weight
                if weight.ndim == 5:  # 3D convolution
                    out_channels, kH, kW, kD, in_channels = weight.shape
                    fan_in = in_channels * kH * kW * kD
                    fan_out = out_channels * kH * kW * kD
                    
                    # Kaiming initialization for ReLU: std = sqrt(2 / fan_out) for mode='fan_out'
                    # We apply a scaling factor of 2 to match ME's effective output magnitude
                    # This accounts for subtle differences in variance propagation through
                    # sparse convolutions and the U-Net architecture and makes sure we have enough outputs above the graspenss threshold
                    std = (2.0 / fan_out) ** 0.5 * 2
                    with torch.no_grad():
                        weight.normal_(0, std)
                else:
                    # Fallback to standard initialization for unexpected shapes
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1, indice_key_prefix=None):
        downsample = None
        out_channels = planes * block.expansion
        key_prefix = indice_key_prefix or "res_stage"

        if stride != 1 or self.inplanes != out_channels:
            downsample = spconv.SparseSequential(
                spconv.SparseConv3d(
                    self.inplanes, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                    indice_key=f"{key_prefix}_down"
                ),
                nn.BatchNorm1d(out_channels, momentum=bn_momentum),
            )

        layers = []
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                bn_momentum=bn_momentum,
                indice_key=f"{key_prefix}_subm"
            )
        )
        self.inplanes = out_channels

        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=1,
                    dilation=dilation,
                    downsample=None,
                    bn_momentum=bn_momentum,
                    indice_key=f"{key_prefix}_subm"
                )
            )

        return spconv.SparseSequential(*layers)

    def forward(self, x: spconv.SparseConv3d):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.glob_pool(x)
        return self.final(x)
