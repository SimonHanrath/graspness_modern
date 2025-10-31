# overfit_one_batch_spconv_unet14d.py
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import math
import random

from models.backbone_resunet14 import SPconvUNet14D


def make_random_sparse_input(B=2, N=15000, C_in=3, spatial_shape=(48,48,48), device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator().manual_seed(0)
    b = torch.randint(0, B, (N,1), dtype=torch.int32, generator=rng)
    x = torch.randint(0, spatial_shape[0], (N,1), dtype=torch.int32, generator=rng)
    y = torch.randint(0, spatial_shape[1], (N,1), dtype=torch.int32, generator=rng)
    z = torch.randint(0, spatial_shape[2], (N,1), dtype=torch.int32, generator=rng)
    coords = torch.cat([b,x,y,z], dim=1)
    # de-dup to avoid collisions
    coords = torch.unique(coords, dim=0)
    N_eff = coords.shape[0]
    feats = torch.randn(N_eff, C_in, device=device)
    coords = coords.to(device)
    sp_input = spconv.SparseConvTensor(feats, coords, spatial_shape, B)
    return sp_input

class VoxelHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls = spconv.SubMConv3d(in_channels, num_classes, kernel_size=1, bias=True, indice_key="head")
    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        return self.cls(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C_in, num_classes = 2, 3, 8
    spatial_shape = (48, 48, 48)
    seed_feat_dim = 128

    # data
    sp_input = make_random_sparse_input(B=B, N=20000, C_in=C_in, spatial_shape=spatial_shape, device=device)

    # model
    backbone = SPconvUNet14D(in_channels=C_in, out_channels=seed_feat_dim, D=3).to(device)
    head = VoxelHead(in_channels=seed_feat_dim, num_classes=num_classes).to(device)

    # targets (random voxel-wise labels on active sites of head input)
    backbone.train(); head.train()
    optim = torch.optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # training loop
    steps = 200
    best = math.inf
    for it in range(1, steps + 1):
        optim.zero_grad(set_to_none=True)
        x = sp_input
        x = backbone(x)
        x = backbone.final(x)
        x = x.replace_feature(backbone.relu(x.features))  # keep SparseConvTensor + indice_dict intact

        x = head(x)   # now head.in_channels should be seed_feat_dim

        logits = x.features             # [N_active, num_classes]
        N_active = logits.size(0)
        # make a fixed random target per active voxel to overfit
        torch.manual_seed(42)  # keep fixed targets across steps
        target = torch.randint(0, num_classes, (N_active,), device=logits.device)

        loss = criterion(logits, target)
        loss.backward()

        with torch.no_grad():
            total = 0
            finite = 0
            for p in list(backbone.parameters()) + list(head.parameters()):
                if p.grad is not None:
                    total += 1
                    if torch.isfinite(p.grad).all():
                        finite += 1
            assert total > 0 and finite == total, "Found missing or non-finite gradients."


        torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), 1.0)
        optim.step()

        best = min(best, loss.item())
        if it % 20 == 0 or it == 1:
            print(f"step {it:03d} | loss {loss.item():.4f} | best {best:.4f}")

    print("✓ Overfit-one-batch run complete.")

if __name__ == "__main__":
    main()
