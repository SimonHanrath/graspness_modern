import numpy as np
import os
from PIL import Image
import scipy.io as scio
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image
from utils.knn_utils import knn_query
import torch
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import get_obj_pose_list, transform_points
import argparse
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, default=None)
parser.add_argument('--camera_type', default='kinect', help='[realsense/kinect]')

# Output-safe speed/resume controls (defaults preserve original behavior)
parser.add_argument('--scene_start', type=int, default=0, help='inclusive')
parser.add_argument('--scene_end',   type=int, default=100, help='exclusive; 100(original) or 190(full)')
parser.add_argument('--ann_start',   type=int, default=0, help='inclusive')
parser.add_argument('--ann_end',     type=int, default=256, help='exclusive; 256 original')
parser.add_argument('--chunk',       type=int, default=10000, help='points per KNN batch (10k original)')

if __name__ == '__main__':
    cfgs = parser.parse_args()
    dataset_root = cfgs.dataset_root
    camera_type  = cfgs.camera_type
    save_path_root = os.path.join(dataset_root, 'graspness')

    num_views, num_angles, num_depths = 300, 12, 4
    fric_coef_thresh = 0.8
    point_grasp_num = num_views * num_angles * num_depths

    # OUTER: scenes
    for scene_id in tqdm(range(cfgs.scene_start, cfgs.scene_end), desc='Scenes', position=0):
        save_path = os.path.join(save_path_root, 'scene_' + str(scene_id).zfill(4), camera_type)
        os.makedirs(save_path, exist_ok=True)

        labels = np.load(
            os.path.join(dataset_root, 'collision_label', 'scene_' + str(scene_id).zfill(4), 'collision_labels.npz'))
        collision_dump = [labels[f'arr_{j}'] for j in range(len(labels))]

        # INNER: per-annotation frames
        ann_iter = tqdm(range(cfgs.ann_start, cfgs.ann_end), desc=f'scene {scene_id:04d}', position=1, leave=False)
        for ann_id in ann_iter:
            # get scene point cloud
            depth = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                     camera_type, 'depth', str(ann_id).zfill(4) + '.png')))
            seg = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                   camera_type, 'label', str(ann_id).zfill(4) + '.png')))
            meta = scio.loadmat(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                             camera_type, 'meta', str(ann_id).zfill(4) + '.mat'))
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
            camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                                factor_depth)
            cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

            # remove outlier and get objectness label
            depth_mask = (depth > 0)
            camera_poses = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                camera_type, 'camera_poses.npy'))
            camera_pose = camera_poses[ann_id]
            align_mat = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                             camera_type, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_pose)
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
            cloud_masked = cloud[mask]
            objectness_label = seg[mask]  # kept for parity, not used later

            # get scene object and grasp info
            scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                  camera_type, 'annotations', '%04d.xml' % ann_id))
            pose_vectors = scene_reader.getposevectorlist()
            obj_list, pose_list = get_obj_pose_list(camera_pose, pose_vectors)
            grasp_labels = {}
            for i in obj_list:
                file = np.load(os.path.join(dataset_root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
                grasp_labels[i] = (file['points'].astype(np.float32), file['offsets'].astype(np.float32),
                                   file['scores'].astype(np.float32))

            grasp_points = []
            grasp_points_graspness = []
            for i, (obj_idx, trans_) in enumerate(zip(obj_list, pose_list)):
                sampled_points, offsets, fric_coefs = grasp_labels[obj_idx]
                collision = collision_dump[i]  # Npoints * num_views * num_angles * num_depths
                num_points = sampled_points.shape[0]

                valid_grasp_mask = ((fric_coefs <= fric_coef_thresh) & (fric_coefs > 0) & ~collision)
                valid_grasp_mask = valid_grasp_mask.reshape(num_points, -1)
                graspness = np.sum(valid_grasp_mask, axis=1) / point_grasp_num
                target_points = transform_points(sampled_points, trans_)
                target_points = transform_points(target_points, np.linalg.inv(camera_pose))  # fix bug
                grasp_points.append(target_points)
                grasp_points_graspness.append(graspness.reshape(num_points, 1))
            grasp_points = np.vstack(grasp_points) if len(grasp_points) else np.zeros((0,3), dtype=np.float32)
            grasp_points_graspness = np.vstack(grasp_points_graspness) if len(grasp_points_graspness) else np.zeros((0,1), dtype=np.float32)

            masked_points_num = cloud_masked.shape[0]
            cloud_masked_graspness = np.zeros((masked_points_num, 1), dtype=np.float32)

            # Skip KNN if no grasp points available (empty scene or no valid objects)
            if grasp_points.shape[0] > 0:
                # ---- speed-safe block (no autograd, chunked KNN) ----
                with torch.no_grad():
                    # to GPU - grasp_points: (Np, 3), cloud_masked: (Q, 3)
                    grasp_points_t = torch.from_numpy(grasp_points).cuda()  # (Np, 3)
                    grasp_points_graspness_t = torch.from_numpy(grasp_points_graspness).cuda()  # (Np, 1)

                    CHUNK = int(cfgs.chunk)
                    part_num = int(masked_points_num / CHUNK)
                    for i in range(1, part_num + 2):   # lack of cuda memory
                        if i == part_num + 1:
                            cloud_masked_partial = cloud_masked[CHUNK * part_num:]
                            if len(cloud_masked_partial) == 0:
                                break
                        else:
                            cloud_masked_partial = cloud_masked[CHUNK * (i - 1):(i * CHUNK)]
                        cloud_masked_partial_t = torch.from_numpy(cloud_masked_partial).cuda()  # (chunk, 3)
                        # knn_query: find nearest grasp point for each cloud point
                        # Returns (chunk, 1) -> squeeze to (chunk,)
                        nn_inds = knn_query(grasp_points_t, k=1, query_pos=cloud_masked_partial_t).squeeze(-1)
                        # Clamp indices to valid range as safety measure
                        nn_inds = torch.clamp(nn_inds, 0, grasp_points_graspness_t.shape[0] - 1)
                        cloud_masked_graspness[CHUNK * (i - 1):(i * CHUNK)] = torch.index_select(
                            grasp_points_graspness_t, 0, nn_inds).cpu().numpy()
                # -----------------------------------------------------

            max_graspness = np.max(cloud_masked_graspness) if cloud_masked_graspness.size else 1.0
            min_graspness = np.min(cloud_masked_graspness) if cloud_masked_graspness.size else 0.0
            denom = (max_graspness - min_graspness) if (max_graspness > min_graspness) else 1.0
            cloud_masked_graspness = (cloud_masked_graspness - min_graspness) / denom

            np.save(os.path.join(save_path, str(ann_id).zfill(4) + '.npy'), cloud_masked_graspness)
