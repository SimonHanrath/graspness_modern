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
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=-1,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--vis', action='store_true', default=False)
parser.add_argument('--use_compile', action='store_true', default=False, help='Use torch.compile for inference optimization [PyTorch 2.0+]')
parser.add_argument('--scene', type=str, default='0188')
parser.add_argument('--index', type=str, default='0000')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)


def data_process():
    root = cfgs.dataset_root
    camera_type = cfgs.camera

    depth = np.array(Image.open(os.path.join(root, 'scenes', scene_id, camera_type, 'depth', index + '.png')))
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

    # sample points random
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]

    # Shift so all coords are >= 0 (CRITICAL: must match training preprocessing!)
    offset = -cloud_sampled.min(axis=0)  # [3,]
    cloud_sampled = cloud_sampled + offset

    ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
                'feats': np.ones_like(cloud_sampled).astype(np.float32),
                'cloud_offset': offset.astype(np.float32),  # Store offset to transform back to camera coords
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
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

    net.eval()
    
    # Apply torch.compile for inference optimization (PyTorch 2.0+)
    if cfgs.use_compile:
        print("Compiling model with torch.compile (mode='reduce-overhead')...")
        net = torch.compile(net, mode='reduce-overhead')  # 'reduce-overhead' optimizes for repeated inference
        print("Model compilation enabled.")
    
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
        
        # Debug: Check xyz_graspable coordinates
        if 'xyz_graspable' in end_points:
            xyz_grasp = end_points['xyz_graspable'][0].cpu().numpy()
            print(f"\n=== DEBUG: xyz_graspable (before pred_decode) ===")
            print(f"Range: X=[{xyz_grasp[:, 0].min():.3f}, {xyz_grasp[:, 0].max():.3f}], "
                  f"Y=[{xyz_grasp[:, 1].min():.3f}, {xyz_grasp[:, 1].max():.3f}], "
                  f"Z=[{xyz_grasp[:, 2].min():.3f}, {xyz_grasp[:, 2].max():.3f}]")
        
        grasp_preds = pred_decode(end_points)

    preds = grasp_preds[0].detach().cpu().numpy()
    
    # Debug: Check decoded predictions
    print(f"=== DEBUG: Decoded predictions ===")
    print(f"Grasp centers (col 13:16) range: X=[{preds[:, 13].min():.3f}, {preds[:, 13].max():.3f}], "
          f"Y=[{preds[:, 14].min():.3f}, {preds[:, 14].max():.3f}], "
          f"Z=[{preds[:, 15].min():.3f}, {preds[:, 15].max():.3f}]")
    print("=====================================\n")
    
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
        print(f"Saved offset to: {offset_path}")
        print(f"NOTE: Grasps are in shifted coordinates (all >= 0).")
        print(f"      To transform to camera coordinates: grasp_center -= offset")

    toc = time.time()
    print('inference time: %fs' % (toc - tic))


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
        
        # Debug: Print coordinate ranges
        print(f"\n=== Coordinate Debug Info ===")
        print(f"Point cloud range: X=[{pc[:, 0].min():.3f}, {pc[:, 0].max():.3f}], "
              f"Y=[{pc[:, 1].min():.3f}, {pc[:, 1].max():.3f}], "
              f"Z=[{pc[:, 2].min():.3f}, {pc[:, 2].max():.3f}]")
        if len(gg) > 0:
            grasp_centers = np.array([g.translation for g in gg])
            print(f"Grasp centers range: X=[{grasp_centers[:, 0].min():.3f}, {grasp_centers[:, 0].max():.3f}], "
                  f"Y=[{grasp_centers[:, 1].min():.3f}, {grasp_centers[:, 1].max():.3f}], "
                  f"Z=[{grasp_centers[:, 2].min():.3f}, {grasp_centers[:, 2].max():.3f}]")
            print(f"Total grasps before NMS: {len(gg)}")
        
        gg = gg.nms()
        gg = gg.sort_by_score()
        print(f"Total grasps after NMS: {len(gg)}")
        if gg.__len__() > 30:
            gg = gg[:30]
        print(f"Showing top {len(gg)} grasps")
        print("=============================\n")
        
        grippers = gg.to_open3d_geometry_list()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
        
        # Save visualization to file instead of interactive display
        vis_dir = os.path.join(cfgs.dump_dir, scene_id, cfgs.camera)
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        
        # Option 1: Save point cloud and grasps separately
        pcd_path = os.path.join(vis_dir, f'{cfgs.index}_cloud.ply')
        o3d.io.write_point_cloud(pcd_path, cloud)
        print(f"Point cloud saved to: {pcd_path}")
        
        # Option 2: Use matplotlib for visualization (works in headless environments)
        print("Generating matplotlib-based visualization...")
        try:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot point cloud (downsample for performance)
            pc_sample = pc[::10]
            ax.scatter(pc_sample[:, 0], pc_sample[:, 1], pc_sample[:, 2], 
                      c='gray', marker='.', s=1, alpha=0.3, label='Point Cloud')
            
            # Plot grasp centers and orientations
            num_grasps_to_show = min(10, len(gg))
            for idx in range(num_grasps_to_show):
                g = gg[idx]
                center = g.translation
                # Draw grasp center
                ax.scatter(center[0], center[1], center[2], 
                         c='red', marker='o', s=50, alpha=0.8)
                # Draw approach direction
                approach = g.rotation_matrix[:, 2] * 0.05  # 5cm arrow
                ax.quiver(center[0], center[1], center[2],
                         approach[0], approach[1], approach[2],
                         color='blue', arrow_length_ratio=0.3, linewidth=2)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Grasp Visualization - Scene {cfgs.scene}, Frame {cfgs.index}\nTop {num_grasps_to_show} grasps shown')
            ax.legend()
            
            # Set equal aspect ratio
            max_range = np.array([pc[:, 0].max()-pc[:, 0].min(),
                                 pc[:, 1].max()-pc[:, 1].min(),
                                 pc[:, 2].max()-pc[:, 2].min()]).max() / 2.0
            mid_x = (pc[:, 0].max()+pc[:, 0].min()) * 0.5
            mid_y = (pc[:, 1].max()+pc[:, 1].min()) * 0.5
            mid_z = (pc[:, 2].max()+pc[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Save figure
            fig_path = os.path.join(vis_dir, f'{cfgs.index}_visualization.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Visualization saved to: {fig_path}")
            print(f"Total grasps detected: {len(gg)}")
        except Exception as e:
            print(f"Matplotlib visualization failed: {e}")
            print("Point cloud PLY file saved. View it with MeshLab, CloudCompare, or Open3D on a machine with display.")
        
        # Note: Open3D interactive visualization requires display (X11/OpenGL)
        # To view interactively, run on a machine with display:
        # o3d.visualization.draw_geometries([cloud, *grippers])

        # # Example code for execution
        # g = gg[0]
        # translation = g.translation
        # rotation = g.rotation_matrix

        # pose = translation_rotation_2_matrix(translation,rotation) #transform into 4x4 matrix, should be easy
        # # Transform the grasp pose from camera frame to robot coordinate, implement according to your robot configuration
        # tcp_pose = Camera_To_Robot(pose)

        
        # tcp_ready_pose = copy.deepcopy(tcp_pose)
        # tcp_ready_pose[:3, 3] = tcp_ready_pose[:3, 3] - 0.1 * tcp_ready_pose[:3, 2] # The ready pose is backward along the actual grasp pose by 10cm to avoid collision
       
        # tcp_away_pose = copy.deepcopy(tcp_pose)
        
        # # to avoid the gripper rotate around the z_{tcp} axis in the clock-wise direction.
        # tcp_away_pose[3,:3] = np.array([0,0,-1], dtype=np.float64)
        
        # # to avoid the object collide with the scene.
        # tcp_away_pose[2,3] += 0.1

        # # We rely on python-urx to send the tcp pose the ur5 arm, the package is available at https://github.com/SintefManufacturing/python-urx
        # urx.movels([tcp_ready_pose, tcp_pose], acc = acc, vel = vel, radius = 0.05)

        # # CLOSE_GRIPPER(), implement according to your robot configuration
        # urx.movels([tcp_away_pose, self.throw_pose()], acc = 1.2 * acc, vel = 1.2 * vel, radius = 0.05, wait=False)

