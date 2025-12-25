# TODO: this is currently broken as we removed open3d from the container but here we still require it

import os
import sys
import numpy as np
import argparse
from PIL import Image
import time
import scipy.io as scio
import torch
import open3d as o3d
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from graspnetAPI.graspnet_eval import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import spconv_collate_fn
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/data/datasets/graspnet')
parser.add_argument('--checkpoint_path', default='/data/zibo/logs/graspness_kn.tar')
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default='/data/zibo/logs/')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=100000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=-1,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--vis', action='store_true', default=False)
parser.add_argument('--scene', type=str, default='0001')
parser.add_argument('--index', type=str, default='0000')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)


def data_process():
    root = cfgs.dataset_root
    camera_type = cfgs.camera

    depth = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'depth', index + '.png')))
    rgb = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'rgb', index + '.png')))
    seg = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'label', index + '.png')))
    meta = scio.loadmat(os.path.join(root, 'scenes', scene_id, camera_type, 'meta', index + '.mat'))
    try:
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
    except Exception as e:
        print(repr(e))
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                        factor_depth)
    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    depth_mask = (depth > 0)
    camera_poses = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'camera_poses.npy'))
    align_mat = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'cam0_wrt_table.npy'))
    trans = np.dot(align_mat, camera_poses[int(index)])
    workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
    mask = (depth_mask & workspace_mask)

    cloud_masked = cloud[mask]
    colors_masked = rgb.reshape(-1, 3)[mask.flatten()]

    # sample points random
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    colors_sampled = colors_masked[idxs]

    # Shift so all coords are >= 0 (CRITICAL: must match training preprocessing!)
    offset = -cloud_sampled.min(axis=0)  # [3,]
    cloud_sampled = cloud_sampled + offset

    ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
                'feats': np.ones_like(cloud_sampled).astype(np.float32),
                'cloud_offset': offset.astype(np.float32),  # Store offset to transform back to camera coords
                'cloud_colors': colors_sampled.astype(np.float32) / 255.0,  # Normalize to [0, 1]
                }
    return ret_dict


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def inference(data_input):
    # Enable backend optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    batch_data = spconv_collate_fn([data_input])
    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'], strict = False)
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

    net.eval()
    
    tic = time.time()

    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    # Forward pass - use inference_mode for better performance than no_grad
    with torch.inference_mode():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points)

    preds = grasp_preds[0].detach().cpu().numpy()
    
    # NOTE: Grasps are predicted in the shifted coordinate space (all coords >= 0)
    # For visualization, we keep them in the same space as the point cloud
    # For real-world execution, you would need to transform back to camera coords
    # by subtracting the offset: preds[:, 13:16] = preds[:, 13:16] - offset
    
    # Store offset for potential later use
    if 'cloud_offset' in batch_data:
        offset = batch_data['cloud_offset'][0].cpu().numpy()  # [3,]
        # Save offset with grasps for downstream processing
        # (not applied here to match visualization coordinate frame)
    
    # Filtering grasp poses for real-world execution. 
    # The first mask preserves the grasp poses that are within a 30-degree angle with the vertical pose and have a width of less than 9cm.
    # mask = (preds[:,10] > 0.9) & (preds[:,1] < 0.09)
    # The second mask preserves the grasp poses within the workspace of the robot.
    # workspace_mask = (preds[:,13] > -0.20) & (preds[:,13] < 0.21) & (preds[:,14] > -0.06) & (preds[:,14] < 0.18) & (preds[:,15] > 0.63) 
    # preds = preds[mask & workspace_mask]

    # if len(preds) == 0:
    #         print('No grasp detected after masking')
    #         return

    gg = GraspGroup(preds)
    # collision detection
    if cfgs.collision_thresh > 0:
        cloud = data_input['point_clouds']
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]

    # save grasps
    save_dir = os.path.join(cfgs.dump_dir, scene_id, cfgs.camera)
    save_path = os.path.join(save_dir, cfgs.index + '.npy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gg.save_npy(save_path)
    
    # Also save offset for coordinate transformation back to camera frame
    offset_path = os.path.join(save_dir, cfgs.index + '_offset.npy')
    if 'cloud_offset' in batch_data:
        np.save(offset_path, batch_data['cloud_offset'][0].cpu().numpy())

    toc = time.time()
    print('Inference time: %.2fs' % (toc - tic))


if __name__ == '__main__':
    scene_id = 'scene_' + cfgs.scene
    index = cfgs.index
    data_dict = data_process()

    if cfgs.infer:
        inference(data_dict)
    if cfgs.vis:
        pc = data_dict['point_clouds']
        gg = np.load(os.path.join(cfgs.dump_dir, scene_id, cfgs.camera, cfgs.index + '.npy'))
        gg = GraspGroup(gg)
        
        gg = gg.nms()
        gg = gg.sort_by_score()
        if gg.__len__() > 30:
            gg = gg[:30]
        
        # Generate Open3D gripper geometries
        grippers = gg.to_open3d_geometry_list()
        
        # Save visualization to file
        vis_dir = os.path.join(cfgs.dump_dir, scene_id, cfgs.camera)
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        
        # Create matplotlib visualization with Open3D gripper meshes
        try:
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')
            ax.computed_zorder = False  # Disable automatic z-ordering
            
            # Plot point cloud with RGB colors
            colors = data_dict.get('cloud_colors', np.zeros((len(pc), 3)))
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], 
                      c=colors, marker='.', s=3, alpha=1.0, label='Point Cloud', 
                      zorder=1, depthshade=False)
            
            # Plot grippers using Open3D mesh data
            num_grasps_to_show = min(30, len(grippers))
            top_3_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # Red, Green, Blue
            
            for idx in range(num_grasps_to_show):
                gripper = grippers[idx]
                
                # Color: top 3 are red, green, blue; rest are black
                if idx < 3:
                    color = top_3_colors[idx]
                else:
                    color = [0.0, 0.0, 0.0]  # Black
                
                # Extract mesh vertices and triangles from Open3D geometry
                vertices = np.asarray(gripper.vertices)
                triangles = np.asarray(gripper.triangles)
                
                # Plot the gripper mesh as a surface (drawn on top of point cloud)
                ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                               triangles=triangles, color=color, alpha=1.0,
                               edgecolor='none', shade=True, zorder=10)
            
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_zlabel('Z (m)', fontsize=12)
            ax.set_title(f'Grasp Visualization - Scene {cfgs.scene}, Frame {cfgs.index}\n'
                       f'Showing top {num_grasps_to_show}/{len(gg)} grasps after NMS', 
                       fontsize=14)
            
            # Set equal aspect ratio
            max_range = np.array([pc[:, 0].max()-pc[:, 0].min(),
                                 pc[:, 1].max()-pc[:, 1].min(),
                                 pc[:, 2].max()-pc[:, 2].min()]).max() / 2.0
            mid_x = (pc[:, 0].max()+pc[:, 0].min()) * 0.5
            mid_y = (pc[:, 1].max()+pc[:, 1].min()) * 0.5
            mid_z = (pc[:, 2].max()+pc[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z + max_range, mid_z - max_range)  # Inverted Z-axis
            
            # Set viewing angle - looking in direction of increasing y-axis
            ax.view_init(elev=60, azim=-60)
            
            # Save figure
            fig_path = os.path.join(vis_dir, f'{cfgs.index}_grippers.png')
            plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Visualization saved to: {fig_path}")
            
        except Exception as e:
            print(f"Visualization failed: {e}")

       