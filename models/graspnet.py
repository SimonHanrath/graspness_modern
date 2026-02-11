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
from models.pointcept.backbone_pointnet_transformer_pointcept import (
    PointTransformerV3EncoderFullRes,
    create_ptv3_backbone_grasp,
)
from models.backbone_pointnet2 import PointNet2Backbone

from models.modules import ApproachNet, GraspableNet, CloudCrop, SWADNet
from utils.loss_utils import GRASP_MAX_WIDTH, NUM_VIEW, NUM_ANGLE, NUM_DEPTH, GRASPNESS_THRESHOLD, M_POINT
from utils.label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix
from utils.pointnet.pointnet2_utils import furthest_point_sample, gather_operation, random_points_sample


class GraspNet(nn.Module):
    def __init__(
        self,
        cylinder_radius=0.05,
        seed_feat_dim=512,
        is_training=True,
        backbone='resunet',
        ptv3_pretrained_path=None,
        enable_flash=False,
        enable_stable_score=False,
    ):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = NUM_DEPTH
        self.num_angle = NUM_ANGLE
        self.M_points = M_POINT
        self.num_view = NUM_VIEW
        self.enable_stable_score = enable_stable_score
        self.backbone_type = backbone  # Store backbone type string for forward pass logic
        
        if backbone == 'resunet':
            self.backbone = SPconvUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        elif backbone == 'resunet_rgb':
            self.backbone = SPconvUNet14D(in_channels=6, out_channels=self.seed_feature_dim, D=3)
        elif backbone == 'pointnet2':
            self.backbone = PointNet2Backbone(in_channels=3, out_channels=self.seed_feature_dim)
        elif backbone == 'transformer_pretrained':
            # PTv3 pretrained model expects 6-ch input (XYZ + RGB/normals)
            self.backbone = create_ptv3_backbone_grasp(
                checkpoint_path=ptv3_pretrained_path,
                in_channels=6, 
                out_channels=self.seed_feature_dim,
                use_pretrained=True,
                enable_flash=enable_flash,
            )
        else:  # transformer (default without pretrained)
            self.backbone = PointTransformerV3EncoderFullRes(
                in_channels=3, 
                out_channels=self.seed_feature_dim,
                enc_depths=(1, 1, 1, 2, 1),
                enc_channels=(32, 64, 128, 256, 256),
                enc_num_head=(2, 4, 8, 16, 16),
                enc_patch_size=(64, 64, 64, 64, 64),
                stride=(2, 2, 2, 2),
                dec_depths=(1, 1, 1, 1),
                dec_channels=(64, 96, 128, 256),
                dec_num_head=(4, 6, 8, 16),
                dec_patch_size=(64, 64, 64, 64),
                drop_path=0.1,
                enable_flash=enable_flash,
            )
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        self.rotation = ApproachNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.crop = CloudCrop(nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)
        self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth, enable_stable_score=enable_stable_score)

    def forward(self, end_points):
        seed_xyz = end_points['point_clouds']              
        B, point_num, _ = seed_xyz.shape

        coords = end_points['coors'].to(dtype=torch.int32)  # (M, 4) unique voxel coords in format [batch, x, y, z]
        feats  = end_points['feats'] # (M, Cin) voxelized features
        quantize2original = end_points['quantize2original']  # (N,) mapping from original points to voxels (N total points, values in [0, M))

         # We need to normalize batch-level for proper spatial shape calculation, since spconv expects all cords to be non-negative
        mins = coords[:, 1:].amin(dim=0)          # (3,) [min_x, min_y, min_z]
        maxs = coords[:, 1:].amax(dim=0)          # (3,) [max_x, max_y, max_z]

        coords[:, 1:] = coords[:, 1:] - mins.unsqueeze(0)
        
        extent = (maxs - mins + 1)
        
        # Ensure minimum spatial shape to handle 4 stride 2 layers (downsample by 16)
        # TODO: Without this, flat point clouds (e.g. Z=1) would cause spatial shape reach zero error
        MIN_SPATIAL_DIM = 16
        
        spatial_shape_xyz = (
            max(int(extent[0].item()), MIN_SPATIAL_DIM),  
            max(int(extent[1].item()), MIN_SPATIAL_DIM),  
            max(int(extent[2].item()), MIN_SPATIAL_DIM),  
        )

        coords_bxyz = coords.contiguous().to(torch.int32)

        
        # 6-channel input: [x, y, z, r, g, b] where all are normalized
        if (self.backbone_type in ['transformer_pretrained', 'resunet_rgb']) and (feats.shape[1] == 3):
            
            coord_float = coords[:, 1:].float()  # (M, 3)
            coord_min = coord_float.min(dim=0, keepdim=True).values
            coord_max = coord_float.max(dim=0, keepdim=True).values
            coord_normalized = (coord_float - coord_min) / (coord_max - coord_min + 1e-6)

            feats_6ch = torch.cat([coord_normalized, feats], dim=1)  # (M, 6)
        else:
            feats_6ch = feats 


        sparse_input = spconv.SparseConvTensor(
            feats_6ch,             
            coords_bxyz,           
            spatial_shape_xyz,
            B                      
        )
        
        sparse_output = self.backbone(sparse_input)
        voxel_features = sparse_output.features  # (M, C)
        
        # Map voxel features back to original dense point cloud
        # This is to mimic the approach from MinkowskiEngine
        seed_features = voxel_features[quantize2original].view(B, point_num, -1).transpose(1, 2)  # (B, C, N)

        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim
        objectness_score = end_points['objectness_score']
        graspness_score = end_points['graspness_score'].squeeze(1)
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = graspness_score > GRASPNESS_THRESHOLD
        graspable_mask = objectness_mask & graspness_mask


        seed_features_graspable = []
        seed_xyz_graspable = []
        graspable_num_batch = 0.


        for i in range(B):

            # Filter out unlikely graspable points
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim
            cur_seed_xyz = seed_xyz[i][cur_mask]  # Ns*3
            Ns = cur_seed_xyz.shape[0]
          
            cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # 1*Ns*3
            
            # Handle case where we have fewer graspable points than M_points
            num_to_sample = min(Ns, self.M_points)
            
            if num_to_sample > 0:
                fps_idxs = furthest_point_sample(cur_seed_xyz, num_to_sample) # TODO: test the replacement below
                #fps_idxs = random_points_sample(cur_seed_xyz, num_to_sample)
                cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*Ns
                cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # num_to_sample*3
                cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1*feat_dim*Ns
                cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous() # feat_dim*num_to_sample
            else:
                # use zeros with correct feature dimension
                cur_seed_xyz = torch.zeros((0, 3), device=seed_xyz.device, dtype=seed_xyz.dtype)
                cur_feat = torch.zeros((self.seed_feature_dim, 0), device=seed_features.device, dtype=seed_features.dtype)
            
            # Pad to M_points if necessary
            if num_to_sample < self.M_points:
                pad_num = self.M_points - num_to_sample
                xyz_pad = torch.zeros((pad_num, 3), device=seed_xyz.device, dtype=seed_xyz.dtype)
                cur_seed_xyz = torch.cat([cur_seed_xyz, xyz_pad], dim=0)
                feat_dim = cur_feat.shape[0] if cur_feat.shape[0] > 0 else self.seed_feature_dim
                feat_pad = torch.zeros((feat_dim, pad_num), device=seed_features.device, dtype=seed_features.dtype)
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


def pred_decode(end_points, use_stable_score=False):
    """
    Decode grasp predictions into grasp poses.
    
    Args:
        end_points: dict containing model outputs
        use_stable_score: if True, reweight grasp scores by (1 - stable_score[rot])
    
    Returns:
        List of grasp predictions per batch, each with shape [M, 17]:
        [score, width, height, depth, rotation(9), translation(3), obj_id]
        where M = min(M_POINT, #graspable_points) is determined dynamically
    """
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    
    # Check if stable score predictions are available
    has_stable = 'grasp_stable_pred' in end_points and use_stable_score
    
    for i in range(batch_size):
        grasp_center = end_points['xyz_graspable'][i].float()

        grasp_score = end_points['grasp_score_pred'][i].float()
        # Use dynamic M based on actual tensor size (supports variable M_POINT)
        M = grasp_score.shape[0]
        grasp_score = grasp_score.view(M, NUM_ANGLE*NUM_DEPTH)
        
        if has_stable:
            stable_score = end_points['grasp_stable_pred'][i].float()
            # Expand stable score to match angle*depth shape 
            stable_expanded = stable_score.unsqueeze(-1).expand(-1, -1, NUM_DEPTH) 
            stable_expanded = stable_expanded.reshape(M, NUM_ANGLE*NUM_DEPTH)
            # Reweight
            grasp_score_reweighted = grasp_score * (1.0 - stable_expanded)
            # Find best grasp using reweighted scores
            _, grasp_score_inds = torch.max(grasp_score_reweighted, -1)  
            # Get the original score for the selected grasp (for output compatibility)
            grasp_score_final = torch.gather(grasp_score, 1, grasp_score_inds.view(-1, 1))
        else:
            grasp_score_final, grasp_score_inds = torch.max(grasp_score, -1)
            grasp_score_final = grasp_score_final.view(-1, 1)
        
        grasp_angle = (grasp_score_inds // NUM_DEPTH).float() * (torch.pi / 12)
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1).float() * 0.01
        grasp_depth = grasp_depth.view(-1, 1)
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
        grasp_width = grasp_width.view(M, NUM_ANGLE*NUM_DEPTH)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)

        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(M, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score_final)
        obj_ids = -1 * torch.ones_like(grasp_score_final)
        grasp_preds.append(
            torch.cat([grasp_score_final, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
    return grasp_preds 