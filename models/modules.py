import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import utils.pointnet.pytorch_utils as pt_utils
from utils.pointnet.pointnet2_utils import CylinderQueryAndGroup
from utils.loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix


class GraspableNet(nn.Module): # Objectness and Graspability MLP
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv_graspable = nn.Conv1d(self.in_dim, 3, 1)

    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(seed_features)  # (B, 3, num_seed)
        end_points['objectness_score'] = graspable_score[:, :2]
        end_points['graspness_score'] = graspable_score[:, 2]
        return end_points

class ApproachNet(nn.Module):  # Probabilistic view selection
    def __init__(self, num_view, seed_feature_dim, is_training=True):
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.is_training = is_training

        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)

        # Precompute template views once and store as buffer (moves with .to(device))
        template_views = generate_grasp_views(self.num_view) 
        self.register_buffer("template_views", template_views.float())

    def forward(self, seed_features, end_points):
        B, _, num_seed = seed_features.size()
        res_features = F.relu(self.conv1(seed_features))
        features = self.conv2(res_features)
        view_score = features.transpose(1, 2).contiguous()  # (B, num_seed, num_view)
        end_points['view_score'] = view_score

        if self.is_training:
            view_score_ = view_score.detach()
            view_score_max, _ = view_score_.max(dim=2, keepdim=True)  # (B, num_seed, 1)
            view_score_min, _ = view_score_.min(dim=2, keepdim=True)  # (B, num_seed, 1)
            view_score_ = (view_score_ - view_score_min) / (view_score_max - view_score_min + 1e-8)

            probs_flat = view_score_.view(-1, self.num_view)
            top_view_inds_flat = torch.multinomial(probs_flat, 1, replacement=False)  # (B * num_seed, 1)
            top_view_inds = top_view_inds_flat.view(B, num_seed)  # (B, num_seed)
        else:
            _, top_view_inds = view_score.max(dim=2)  # (B, num_seed)

            template_views = self.template_views  # (num_view, 3)

            top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, 1, 3)
            template_views_expand = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1)
            vp_xyz = torch.gather(template_views_expand, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)

            vp_xyz_ = vp_xyz.view(-1, 3)
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz_.dtype, device=vp_xyz_.device)
            vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)

            end_points['grasp_top_view_xyz'] = vp_xyz
            end_points['grasp_top_view_rot'] = vp_rot

        end_points['grasp_top_view_inds'] = top_view_inds
        return end_points, res_features



class CloudCrop(nn.Module): # Cylinder Grouping
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.04):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3

        self.grouper = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
                                             use_xyz=True, normalize_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, vp_rot,
                                       seed_features_graspable)  # B*3 + feat_dim*M*K
        new_features = self.mlps(grouped_feature)  # (batch_size, mlps[-1], M, K)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (batch_size, mlps[-1], M, 1)
        new_features = new_features.squeeze(-1)   # (batch_size, mlps[-1], M)
        return new_features


class SWADNet(nn.Module): # Grasp pose predicition head
    def __init__(self, num_angle, num_depth, enable_stable_score=False):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.enable_stable_score = enable_stable_score

        self.conv1 = nn.Conv1d(256, 256, 1)  # input feat dim need to be consistent with CloudCrop module
        self.conv_swad = nn.Conv1d(256, 2*num_angle*num_depth, 1)  # scores + widths (96 outputs)
        
        # Separate stable score head - can be frozen/unfrozen independently
        if self.enable_stable_score:
            # Simple linear projection for stable scores (no extra hidden layers)
            # Init with small weights so stable_score starts near 0.5 (neutral)
            self.conv_stable = nn.Conv1d(256, num_angle, 1)
            nn.init.zeros_(self.conv_stable.weight)
            nn.init.zeros_(self.conv_stable.bias)

    def forward(self, vp_features, end_points):
        B, _, num_seed = vp_features.size()
        vp_features_relu = F.relu(self.conv1(vp_features))
        vp_out = self.conv_swad(vp_features_relu)
        vp_out = vp_out.view(B, 2, self.num_angle, self.num_depth, num_seed)
        vp_out = vp_out.permute(0, 1, 4, 2, 3)

        end_points['grasp_score_pred'] = vp_out[:, 0]  # B * num_seed * num_angle * num_depth
        end_points['grasp_width_pred'] = vp_out[:, 1]
        
        if self.enable_stable_score:
            stable_raw = self.conv_stable(vp_features_relu)  # (B, num_angle, num_seed)
            stable_raw = stable_raw.permute(0, 2, 1)  # (B, num_seed, num_angle)
            stable_score = torch.sigmoid(stable_raw)  # Normalize to [0, 1]
            end_points['grasp_stable_pred'] = stable_score  # (B, M, 12)
        
        return end_points
