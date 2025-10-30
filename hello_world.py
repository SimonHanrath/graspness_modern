import torch, spconv
import spconv.pytorch as spt
from spconv.pytorch import SparseConvTensor
import sys

print('Python', sys.version,'Torch', torch.__version__, 'CUDA', torch.version.cuda, 'spconv', spconv.__version__)
coords = torch.tensor([[0,0,0,0],[0,1,1,1]], dtype=torch.int32, device='cuda')
feats  = torch.randn(2, 8, device='cuda')
x = SparseConvTensor(feats, coords, spatial_shape=[2,2,2], batch_size=1)
y = spt.SubMConv3d(8,16,kernel_size=3,padding=1,bias=False).cuda()(x)
print('OK:', y.features.shape)

print(':)')