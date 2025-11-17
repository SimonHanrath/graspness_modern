""" GraspNet baseline model definition.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import spconv.pytorch as spconv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from models.backbone_resunet14 import SPconvUNet14D
from models.modules import ApproachNet, GraspableNet, CloudCrop, SWADNet
from loss_utils import GRASP_MAX_WIDTH, NUM_VIEW, NUM_ANGLE, NUM_DEPTH, GRASPNESS_THRESHOLD, M_POINT
from label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix
from pointnet_replacement.pointnet2_utils import furthest_point_sample, gather_operation


class GraspNet(nn.Module):
    def __init__(self, cylinder_radius=0.05, seed_feat_dim=512, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = NUM_DEPTH
        self.num_angle = NUM_ANGLE
        self.M_points = M_POINT
        self.num_view = NUM_VIEW

        self.backbone = SPconvUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        self.rotation = ApproachNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.crop = CloudCrop(nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)
        self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth)

    def forward(self, end_points):
        seed_xyz = end_points['point_clouds']              
        B, point_num, _ = seed_xyz.shape

        coords = end_points['coors'].to(dtype=torch.int32)  # (M, 4) unique voxel coords in format [batch, x, y, z]
        feats  = end_points['feats'] # (M, Cin) voxelized features
        quantize2original = end_points['quantize2original']  # (N,) mapping from original points to voxels (N total points, values in [0, M))

        # coords[:, 1:] are [x, y, z] coordinates
        mins = coords[:, 1:].amin(dim=0)          # (3,) [min_x, min_y, min_z]
        maxs = coords[:, 1:].amax(dim=0)          # (3,) [max_x, max_y, max_z]

        extent = (maxs - mins + 1)                # (3,) in [X, Y, Z]
        
        # spconv expects spatial_shape in (X, Y, Z) order to match coords format [batch, x, y, z]
        spatial_shape_xyz = (
            int(extent[0].item()),  # X
            int(extent[1].item()),  # Y
            int(extent[2].item()),  # Z
        )

        # coords is already in correct format [batch, x, y, z] - no reordering needed
        coords_bxyz = coords.contiguous().to(torch.int32)

        sparse_input = spconv.SparseConvTensor(
            feats,                 # (M, Cin) where M is unique voxels
            coords_bxyz,           # (M, 4) [batch, x, y, z], int32
            spatial_shape_xyz,     # (X, Y, Z)
            B                      # batch size
        )
        
        # spconv UNet will preserve voxel structure (same coordinates, different features)
        sparse_output = self.backbone(sparse_input)
        voxel_features = sparse_output.features  # (M, C) - same M voxels as input
        
        # Map voxel features back to original dense point cloud using quantize2original
        # quantize2original[i] gives the voxel index for original point i
        # This is the same approach as MinkowskiEngine
        seed_features = voxel_features[quantize2original].view(B, point_num, -1).transpose(1, 2)  # (B, C, N)

        """# --- DEBUG: Backbone output analysis ---
        with torch.no_grad():
            mean_val = seed_features.mean().item()
            std_val = seed_features.std().item()
            min_val = seed_features.min().item()
            max_val = seed_features.max().item()
            print(f"[DEBUG] Backbone output stats:")
            print(f"  Shape: {tuple(seed_features.shape)}")
            print(f"  Mean: {mean_val:.6f}, Std: {std_val:.6f}")
            print(f"  Min: {min_val:.6f}, Max: {max_val:.6f}")
            # Optional: visualize histogram for one batch
            if torch.isnan(seed_features).any():
                print("  WARNING: NaNs detected in backbone features!")
        # --- END DEBUG ---"""

        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim
        objectness_score = end_points['objectness_score']
        graspness_score = end_points['graspness_score'].squeeze(1)
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = graspness_score > GRASPNESS_THRESHOLD
        graspable_mask = objectness_mask & graspness_mask

        """#dummy code to get good mask TODO: Fix this, our backbone model seems to produce outputs with lower magnitude and therefore the MLP layer assign low scores and therefore our mask has to few points
        graspable_mask = torch.zeros_like(objectness_mask, dtype=torch.bool)
        min_needed = self.M_points
        for i in range(B):
            # choose at least M_points per sample (or as many as available)
            k = min(point_num, max(min_needed, point_num // 2))  # pick a sensible number
            sel = torch.randperm(point_num, device=graspable_mask.device)[:k]
            graspable_mask[i, sel] = True"""

        seed_features_graspable = []
        seed_xyz_graspable = []
        graspable_num_batch = 0.

        """# --- DEBUG START ---
        # Masks
        print("\n=== DEBUG: Graspable selection block ===")
        print(f"seed_features: {seed_features.shape}")  # (B, feat_dim, Ns)
        print(f"seed_features_flipped: {seed_features_flipped.shape}")  # (B, Ns, feat_dim) — double-check

        # Scores
        print(f"objectness_score: {objectness_score.shape} (min={objectness_score.min():.3f}, max={objectness_score.max():.3f})")
        print(f"graspness_score: {graspness_score.shape} (min={graspness_score.min():.3f}, max={graspness_score.max():.3f})")

        # Masks
        print(f"objectness_pred: {objectness_pred.shape}, unique={objectness_pred.unique(return_counts=True)}")
        print(f"objectness_mask: {objectness_mask.shape}, num_true={objectness_mask.sum().item()}")
        print(f"graspness_mask: {graspness_mask.shape}, num_true={graspness_mask.sum().item()}")
        print(f"graspable_mask: {graspable_mask.shape}, num_true={graspable_mask.sum().item()}")

        # --- DEBUG END ---"""

        for i in range(B):#  TODO: think about why we need to iterate over the batches and not parallelize this

            # Filter out unlikely graspable points
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim
            cur_seed_xyz = seed_xyz[i][cur_mask]  # Ns*3
            Ns = cur_seed_xyz.shape[0]
            #print(f"Graspable num for batch {i}: {cur_mask.sum().item()}")
            #print(F"Ns: {Ns}")
           
            #perform FPS to get fixed number of points
            cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # 1*Ns*3
            
            # Handle case where we have fewer graspable points than M_points
            num_to_sample = min(Ns, self.M_points)
            
            if num_to_sample > 0:
                fps_idxs = furthest_point_sample(cur_seed_xyz, num_to_sample)
                cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*Ns
                cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # num_to_sample*3
                cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1*feat_dim*Ns
                cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous() # feat_dim*num_to_sample
            else:
                # No graspable points - use zeros (this sample will likely be ignored in loss)
                cur_seed_xyz = torch.zeros((0, 3), device=cur_seed_xyz.device, dtype=cur_seed_xyz.dtype)
                cur_feat = torch.zeros((cur_feat.shape[0], 0), device=cur_feat.device, dtype=cur_feat.dtype)
            
            # Pad to M_points if necessary
            if num_to_sample < self.M_points:
                pad_num = self.M_points - num_to_sample
                # Pad xyz with zeros (or repeat last point if you prefer)
                xyz_pad = torch.zeros((pad_num, 3), device=cur_seed_xyz.device, dtype=cur_seed_xyz.dtype)
                cur_seed_xyz = torch.cat([cur_seed_xyz, xyz_pad], dim=0)
                # Pad features with zeros
                feat_pad = torch.zeros((cur_feat.shape[0], pad_num), device=cur_feat.device, dtype=cur_feat.dtype)
                cur_feat = torch.cat([cur_feat, feat_pad], dim=1)

            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)

        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # B*Ns*3
        seed_features_graspable = torch.stack(seed_features_graspable)  # B*feat_dim*Ns
        end_points['xyz_graspable'] = seed_xyz_graspable
        end_points['graspable_count_stage1'] = graspable_num_batch / B

        end_points, res_feat = self.rotation(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat

        if self.is_training:
            end_points = process_grasp_labels(end_points)
            grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points)
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']

        group_features = self.crop(seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        end_points = self.swad(group_features, end_points)

        return end_points


def pred_decode(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        grasp_center = end_points['xyz_graspable'][i].float()

        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_score = grasp_score.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_score, grasp_score_inds = torch.max(grasp_score, -1)  # [M_POINT]
        grasp_score = grasp_score.view(-1, 1)
        grasp_angle = (grasp_score_inds // NUM_DEPTH) * np.pi / 12
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
        grasp_width = grasp_width.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)

        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(M_POINT, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
    return grasp_preds 