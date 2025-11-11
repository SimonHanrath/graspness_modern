import os
import sys
import math
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import spconv.pytorch as spconv

# ---------------- paths like in your train script ----------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.backbone_resunet14 import SPconvUNet14D
from dataset.graspnet_dataset import GraspNetDataset, spconv_collate_fn, load_grasp_labels



def make_sparse_input(end_points):
    seed_xyz = end_points['point_clouds']              
    B, point_num, _ = seed_xyz.shape

    coords = end_points['coors'].to(dtype=torch.int32)  # (N, 4)
    feats  = end_points['feats'] # (N, Cin)

    mins = coords[:, 1:].amin(dim=0)          # (3,) [min_x, min_y, min_z]
    maxs = coords[:, 1:].amax(dim=0)          # (3,) [max_x, max_y, max_z]

    coords[:, 1:] -= mins

    extent = (maxs - mins + 1)                # (3,) in [X, Y, Z]
    spatial_shape_zyx = (
        int(extent[2].item()),  # Z
        int(extent[1].item()),  # Y
        int(extent[0].item()),  # X
        )

    # Reorder to spconv layout: [b, z, y, x]
    coords_bzyx = torch.stack(
        (coords[:, 0], coords[:, 3], coords[:, 2], coords[:, 1]),
        dim=1
    ).contiguous().to(torch.int32)

    #print(f'Spatial shape: {spatial_shape_zyx}')

    sparse_input = spconv.SparseConvTensor(
        feats,                 # (N, Cin)
        coords_bzyx,           # (N, 4) [b, z, y, x]
        spatial_shape_zyx,
        B                      # batch size
    )

    return sparse_input

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

def move_batch_to_device(batch, device):
    """Mirror your train script's device move logic."""
    for key in batch:
        if 'list' in key:
            for i in range(len(batch[key])):
                for j in range(len(batch[key][i])):
                    batch[key][i][j] = batch[key][i][j].to(device)
        else:
            if hasattr(batch[key], "to"):
                batch[key] = batch[key].to(device)
    return batch



class ReconHead(nn.Module):
    """
    1x1 (SubM) sparse conv to map backbone features back to input feature dimension
    at identical active sites.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size=1, bias=True, indice_key="recon"
        )

    def forward(self, x: spconv.SparseConvTensor):
        return self.proj(x)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- dataset (no labels needed) ----------
    # We still need grasp labels util to satisfy dataset init if it expects it.
    # If your GraspNetDataset can handle load_label=False without passing labels, set that.
    dataset_root = "/datasets/graspnet"
    grasp_labels = load_grasp_labels(dataset_root)


    ds = GraspNetDataset(dataset_root, grasp_labels=grasp_labels, camera='kinect', split='train',  
                               num_points=15000, voxel_size=0.005,  remove_outlier=True, augment=True, load_label=True)  
    print('train dataset length: ', len(ds))
    
    dl = DataLoader(ds, batch_size=1, shuffle=True,  num_workers=0,
                                  worker_init_fn=my_worker_init_fn, collate_fn=spconv_collate_fn)
    print('train dataloader length: ', len(dl))

    # ---- warm one batch to detect feature dim ----
    in_channels = 3

    # ---- backbone + recon head ----
    backbone = SPconvUNet14D(in_channels=in_channels, out_channels=512, D=3).to(device)
    head = ReconHead(in_channels=512, out_channels=in_channels).to(device)

    params = list(backbone.parameters()) + list(head.parameters())
    optim_ = optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
   
    global_step = 0
    best = math.inf

    for epoch in range(1, 3):
        backbone.train(); head.train()
        running = 0.0
        for batch_idx, batch_data_label in enumerate(dl): # TODO: here I left of, remove the extract_sparse_input function and make it closer to the train script
            #batch_data_label = move_batch_to_device(batch_data_label, device)
            for key in batch_data_label:
                if 'list' in key:
                    for i in range(len(batch_data_label[key])):
                        for j in range(len(batch_data_label[key][i])):
                            batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
                else:
                    batch_data_label[key] = batch_data_label[key].to(device)


            sparse_input = make_sparse_input(batch_data_label)
            target = sparse_input.features  # [N_active, Cin]

            optim_.zero_grad(set_to_none=True)
            out = head(backbone(sparse_input))  # SparseConvTensor
            pred = out.features  # [N_active, C_in]


            loss = criterion(pred, target)
            loss.backward()

            # gradient health check
            with torch.no_grad():
                total = 0
                finite = 0
                for p in params:
                    if p.grad is not None:
                        total += 1
                        if torch.isfinite(p.grad).all():
                            finite += 1
                assert total > 0 and finite == total, "Found missing or non-finite gradients."

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim_.step()

            running += loss.item()
            best = min(best, loss.item())
            global_step += 1

            if batch_idx % 20 == 0 or batch_idx == 1:
                print(f"[{datetime.now()}] epoch {epoch:02d} | iter {batch_idx:04d} "
                      f"| N_active {pred.size(0)} | loss {loss.item():.6f} | best {best:.6f}")

        epoch_mean = running / max(1, batch_idx)
        print(f"[{datetime.now()}] ---- epoch {epoch:02d} mean loss: {epoch_mean:.6f} ----")

    print("✓ Backbone reconstruction sanity test complete.")


if __name__ == "__main__":
    main()
