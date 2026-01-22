""" GraspNet dataset processing.
    Author: chenxi-wang
"""

import os
import numpy as np
import scipy.io as scio
from PIL import Image
from functools import lru_cache

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, get_workspace_mask


class LazyGraspLabels:
    """
    Lazy loader for grasp labels with LRU caching.
    Can be pickled for multiprocessing (unlike nested classes).
    """
    def __init__(self, root):
        self.root = root
        self.cache = {}
        self.cache_maxsize = 20  # Keep 20 objects in cache (~5GB)
    
    def __getitem__(self, obj_name):
        if obj_name in self.cache:
            return self.cache[obj_name]
        
        label = np.load(os.path.join(self.root, 'grasp_label_simplified', '{}_labels.npz'.format(str(obj_name - 1).zfill(3))))
        result = (label['points'].astype(np.float32), 
                 label['width'].astype(np.float32),
                 label['scores'].astype(np.float32))
        
        # Simple LRU cache management
        if len(self.cache) >= self.cache_maxsize:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        
        self.cache[obj_name] = result
        return result


class GraspNetDataset(Dataset):
    def __init__(self, root, grasp_labels=None, camera='kinect', split='train', num_points=20000,
                 voxel_size=0.005, remove_outlier=True, augment=False, load_label=True, use_rgb=False):
        assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.use_rgb = use_rgb  # Whether to include RGB features (for 6-channel input)
        
        # Cache for collision labels - use LRU cache with max size to limit memory
        # This allows recently accessed scenes to stay in memory while releasing old ones
        self._collision_cache = {}
        self._collision_cache_maxsize = 5  # Keep at most 5 scenes in cache (~900MB)

        if split == 'train':
            self.sceneIds = list(range(0,100))
        elif split == 'val':
            self.sceneIds = list(range(109, 110))
        elif split == 'test':
            self.sceneIds = list(range(110, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(10, 11))  # Match eval_seen which uses scene_0110
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.depthpath = []
        self.rgbpath = []  # RGB image paths
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        for x in tqdm(self.sceneIds, desc='Loading data paths...'):
            for img_num in range(256):
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.rgbpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, str(img_num).zfill(4) + '.npy'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
            # REMOVED: No longer loading all collision labels at initialization
            # This prevents each DataLoader worker from holding 3GB+ of collision data in memory

    def scene_list(self):
        return self.scenename
    
    def _load_collision_labels(self, scene):
        """
        Lazy load collision labels for a specific scene on-demand.
        Uses an LRU cache to keep recently accessed scenes in memory.
        This prevents memory explosion with multiple DataLoader workers.
        """
        if scene in self._collision_cache:
            return self._collision_cache[scene]
        
        # Load collision labels for this scene
        collision_labels_path = os.path.join(self.root, 'collision_label', scene, 'collision_labels.npz')
        collision_labels_npz = np.load(collision_labels_path)
        
        # Convert to dictionary format
        collision_dict = {}
        for i in range(len(collision_labels_npz)):
            collision_dict[i] = collision_labels_npz['arr_{}'.format(i)]
        
        # Implement simple LRU: if cache is full, remove oldest entry
        if len(self._collision_cache) >= self._collision_cache_maxsize:
            # Remove the first (oldest) item
            oldest_scene = next(iter(self._collision_cache))
            del self._collision_cache[oldest_scene]
        
        # Add to cache
        self._collision_cache[scene] = collision_dict
        return collision_dict

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        
        # Load RGB if needed
        if self.use_rgb:
            rgb = np.array(Image.open(self.rgbpath[index]))  # (H, W, 3) uint8
        
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        
        # Apply same mask to RGB
        if self.use_rgb:
            rgb_masked = rgb[mask]  # (N, 3) uint8

        if return_raw_cloud:
            return cloud_masked
        # sample points random
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        
        if self.use_rgb:
            rgb_sampled = rgb_masked[idxs]  # (num_points, 3)

        offset = -cloud_sampled.min(axis=0)  # [3,]
        cloud_sampled = cloud_sampled + offset

        # Features: either ones (3-ch) or normalized RGB (3-ch for 6-ch input: XYZ coords + RGB feats)
        if self.use_rgb:
            # Normalize RGB to [0, 1] range
            feats = rgb_sampled.astype(np.float32) / 255.0
        else:
            feats = np.ones_like(cloud_sampled).astype(np.float32)

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats': feats,
                    'cloud_offset': offset.astype(np.float32),  # Store offset to transform back to camera coords
                    }
        return ret_dict

    def get_data_label(self, index):
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        graspness = np.load(self.graspnesspath[index])  # for each point in workspace masked point cloud
        scene = self.scenename[index]
        
        # Load RGB if needed
        if self.use_rgb:
            rgb = np.array(Image.open(self.rgbpath[index]))  # (H, W, 3) uint8
        
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        seg_masked = seg[mask]
        
        # Apply same mask to RGB
        if self.use_rgb:
            rgb_masked = rgb[mask]  # (N, 3) uint8

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        seg_sampled = seg_masked[idxs]
        graspness_sampled = graspness[idxs]
        objectness_label = seg_sampled.copy()
        
        if self.use_rgb:
            rgb_sampled = rgb_masked[idxs]  # (num_points, 3)

        objectness_label[objectness_label > 1] = 1

        object_poses_list = []
        grasp_points_list = []
        grasp_widths_list = []
        grasp_scores_list = []
        
        # Lazy load collision labels for this scene only when needed
        collision_labels_scene = self._load_collision_labels(scene) if self.load_label else None
        
        for i, obj_idx in enumerate(obj_idxs):
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, widths, scores = self.grasp_labels[obj_idx]
            collision = collision_labels_scene[i]  # (Np, V, A, D)

            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_widths_list.append(widths[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)

        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)

        # Shift so all coords are >= 0
        offset = -cloud_sampled.min(axis=0)  # [3,]
        cloud_sampled = cloud_sampled + offset

        # Features: either ones (3-ch) or normalized RGB (3-ch for 6-ch input: XYZ coords + RGB feats)
        if self.use_rgb:
            # Normalize RGB to [0, 1] range
            feats = rgb_sampled.astype(np.float32) / 255.0
        else:
            feats = np.ones_like(cloud_sampled).astype(np.float32)

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats': feats,
                    'cloud_offset': offset.astype(np.float32),  # Store offset to transform back to camera coords
                    'graspness_label': graspness_sampled.astype(np.float32),
                    'objectness_label': objectness_label.astype(np.int64),
                    'object_poses_list': object_poses_list,
                    'grasp_points_list': grasp_points_list,
                    'grasp_widths_list': grasp_widths_list,
                    'grasp_scores_list': grasp_scores_list}
        return ret_dict


def load_grasp_labels(root):
    """
    Load grasp labels for all objects.
    NOTE: Returns ~21GB of data. This is loaded once in the main process 
    and shared across workers via copy-on-write in fork mode, or needs to be 
    passed to each worker in spawn mode. With lazy loading of collision labels,
    this is now the main memory bottleneck.
    """
    obj_names = list(range(1, 89))
    grasp_labels = {}
    for obj_name in tqdm(obj_names, desc='Loading grasping labels...'):
        label = np.load(os.path.join(root, 'grasp_label_simplified', '{}_labels.npz'.format(str(obj_name - 1).zfill(3))))
        grasp_labels[obj_name] = (label['points'].astype(np.float32), label['width'].astype(np.float32),
                                  label['scores'].astype(np.float32))

    return grasp_labels


def load_grasp_labels_lazy(root):
    """
    Alternative lazy loading for grasp labels.
    Returns a lazy loader object instead of loading all labels upfront.
    Use this if memory is extremely constrained.
    """
    return LazyGraspLabels(root)


def spconv_collate_fn(list_data):
    """
    Collate function for spconv that mimics MinkowskiEngine's sparse_quantize behavior.
    
    Key steps:
    1. Concatenate all point coordinates and features from the batch
    2. Find unique voxel coordinates using torch.unique()
    3. Create quantize2original mapping (like ME.utils.sparse_quantize)
    4. Average features for points that map to the same voxel
    
    This ensures that:
    - Multiple points in the same voxel get averaged features (proper voxelization)
    - We can map voxel features back to original points using quantize2original
    """

    coords_list = []
    feats_list = []
    for b, d in enumerate(list_data):
        c = d["coors"]
        f = d["feats"]

        if not torch.is_tensor(c):
            c = torch.as_tensor(c)
        if not torch.is_tensor(f):
            f = torch.as_tensor(f, dtype=torch.float32)

        c = c.int()  
        bcol = torch.full((c.shape[0], 1), b, dtype=torch.int32, device=c.device)
        coords_list.append(torch.cat([bcol, c], dim=1))  
        feats_list.append(f)

    coordinates_batch = torch.cat(coords_list, dim=0).contiguous()         
    features_batch    = torch.cat(feats_list,  dim=0).contiguous().float() 

    # Create quantize2original mapping: for each original point, which voxel does it map to?
    # Multiple points can map to the same voxel
    # We need to find unique voxel coordinates and create the mapping
    unique_coords, quantize2original = torch.unique(coordinates_batch, dim=0, return_inverse=True)
    
    # Average features for points that map to the same voxel
    num_unique = unique_coords.shape[0]
    voxel_features = torch.zeros((num_unique, features_batch.shape[1]), 
                                  dtype=features_batch.dtype, device=features_batch.device)
    voxel_features.scatter_add_(0, quantize2original.unsqueeze(1).expand(-1, features_batch.shape[1]), 
                                 features_batch)
    
    # Count how many points map to each voxel to compute average
    counts = torch.zeros(num_unique, dtype=torch.float32, device=features_batch.device)
    counts.scatter_add_(0, quantize2original, torch.ones_like(quantize2original, dtype=torch.float32))
    voxel_features = voxel_features / counts.unsqueeze(1).clamp(min=1)
    
    res = {
        "coors": unique_coords,          
        "feats": voxel_features,
        "quantize2original": quantize2original,
    }

    def collate_fn_(batch):
        if isinstance(batch[0], torch.Tensor):
            return torch.stack(batch, 0)
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[(torch.from_numpy(sample) if type(sample).__module__ == 'numpy' else sample)
                     for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key in ('coors', 'feats'):
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
        else:
            return batch

    res = collate_fn_(list_data)
    return res