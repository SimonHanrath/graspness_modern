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
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, get_workspace_mask
from utils.stable_score_utils import ensure_stable_labels_exist, check_stable_labels_available


class SceneAwareSampler(Sampler):
    """
    A sampler that groups samples by scene to maximize cache efficiency.
    
    Instead of randomly sampling across all indices (which causes cache thrashing
    with lazy-loaded collision labels), this sampler:
    1. Shuffles the order of scenes
    2. Shuffles frames within each scene
    3. Returns all frames from one scene before moving to the next
    
    This ensures that collision labels for a scene stay in cache while all
    frames from that scene are being processed.
    """
    def __init__(self, dataset, shuffle=True, seed=None):
        """
        Args:
            dataset: GraspNetDataset instance
            shuffle: If True, shuffle scenes and frames within scenes
            seed: Random seed for reproducibility (set per epoch for distributed training)
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Build scene to indices mapping
        self.scene_to_indices = {}
        for idx, scene in enumerate(dataset.scenename):
            if scene not in self.scene_to_indices:
                self.scene_to_indices[scene] = []
            self.scene_to_indices[scene].append(idx)
        
        self.scenes = list(self.scene_to_indices.keys())
    
    def __iter__(self):
        # Create a generator for reproducibility
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch if self.seed is not None else self.epoch)
            
            # Shuffle scene order
            scene_order = torch.randperm(len(self.scenes), generator=g).tolist()
            
            for scene_idx in scene_order:
                scene = self.scenes[scene_idx]
                indices = self.scene_to_indices[scene].copy()
                # Shuffle frames within scene
                frame_order = torch.randperm(len(indices), generator=g).tolist()
                for i in frame_order:
                    yield indices[i]
        else:
            # No shuffling - iterate in order
            for scene in self.scenes:
                for idx in self.scene_to_indices[scene]:
                    yield idx
    
    def __len__(self):
        return len(self.dataset)
    
    def set_epoch(self, epoch):
        """Set epoch for shuffling reproducibility (important for distributed training)."""
        self.epoch = epoch


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


def sample_floor_aware(cloud, trans, num_points, floor_height=0.01, floor_keep_ratio=0.01):
    """
    Sample points with reduced probability for floor points.
    
    Args:
        cloud: (N, 3) point cloud in camera coordinates
        trans: (4, 4) transformation matrix to table coordinates
        num_points: number of points to sample
        floor_height: maximum height to consider as floor (meters)
        floor_keep_ratio: sampling weight for floor points relative to object points
        
    Returns:
        idxs: indices of sampled points
    """
    n = len(cloud)
    
    # Transform to table coordinates to get height
    ones = np.ones((n, 1))
    cloud_homo = np.hstack([cloud, ones])  # (N, 4)
    cloud_table = (trans @ cloud_homo.T).T[:, :3]  # (N, 3)
    heights = cloud_table[:, 2]  # Z is height in table coords
    
    # Create sampling weights: floor points get reduced probability
    floor_mask = heights < floor_height
    weights = np.ones(n)
    weights[floor_mask] = floor_keep_ratio
    probs = weights / weights.sum()
    
    if n >= num_points:
        idxs = np.random.choice(n, num_points, replace=False, p=probs)
    else:
        idxs1 = np.arange(n)
        idxs2 = np.random.choice(n, num_points - n, replace=True, p=probs)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    
    return idxs


class GraspNetDataset(Dataset):
    def __init__(self, root, grasp_labels=None, camera='kinect', split='train', num_points=20000,
                 voxel_size=0.005, remove_outlier=True, augment=False, load_label=True, use_rgb=False,
                 enable_stable_score=False, floor_sampling=False, max_views=256):
        assert (num_points <= 300000)  # Raised from 50k; adjust based on GPU memory
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.use_rgb = use_rgb  
        self.enable_stable_score = enable_stable_score
        self.floor_sampling = floor_sampling 
        
        # Cache for stable score labels per object
        self._stable_labels_cache = {}
        self._stable_labels_path = os.path.join(root, 'stable_labels')
        
        # Auto-compute stable labels if enabled and missing
        if enable_stable_score:
            all_exist, missing_count = check_stable_labels_available(root)
            if not all_exist:
                print(f"Stable labels missing for {missing_count} objects. Auto-computing...")
                success = ensure_stable_labels_exist(root, verbose=True)
                if not success:
                    raise RuntimeError(
                        "Failed to compute stable labels. Please install trimesh: pip install trimesh"
                    )
        
        # use LRU cache 
        # this allows recently accessed scenes to stay in memory while releasing old ones
        # Size of 10 scenes balances memory ( ca. 1.8GB) with hit rate for batched scene access
        self._collision_cache = {}
        self._collision_cache_maxsize = 10 

        if split == 'train':
            self.sceneIds = list(range(0,100))
        elif split == 'train_all':
            self.sceneIds = list(range(0, 190))  # All 190 scenes for training
        elif split == 'val':
            self.sceneIds = list(range(109, 110))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))  # scenes 110-129 for test_seen evaluation
        elif split == 'test_seen_single':
            self.sceneIds = list(range(181, 182))  # Just scene_0181
        elif split == 'test_seen_mini':
            self.sceneIds = [101, 115, 128]  # Mini test_seen subset
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_similar_mini':
            self.sceneIds = [131, 145, 158]  # Mini test_similar subset
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        elif split == 'test_novel_single':
            self.sceneIds = list(range(180, 181))  # Just scene_0180
        elif split == 'test_novel_mini':
            self.sceneIds = [161, 175, 188]  # Mini test_novel subset
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.depthpath = []
        self.rgbpath = []  # RGB image paths
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        for x in tqdm(self.sceneIds, desc='Loading data paths...'):
            for img_num in range(max_views):
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.rgbpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, str(img_num).zfill(4) + '.npy'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
            
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
        
        collision_labels_path = os.path.join(self.root, 'collision_label', scene, 'collision_labels.npz')
        collision_labels_npz = np.load(collision_labels_path)
        
        # Convert to dictionary format
        collision_dict = {}
        for i in range(len(collision_labels_npz)):
            collision_dict[i] = collision_labels_npz['arr_{}'.format(i)]
        
        if len(self._collision_cache) >= self._collision_cache_maxsize:
            oldest_scene = next(iter(self._collision_cache))
            del self._collision_cache[oldest_scene]
        
        self._collision_cache[scene] = collision_dict
        return collision_dict
    
    def _load_stable_labels(self, obj_idx): # TODO: Verify this for correctness
        """
        Lazy load stable score labels for a specific object on-demand.
        
        Args:
            obj_idx: 1-indexed object ID (1-88)
        
        Returns:
            stable_labels: np.ndarray of shape (Np, V, A) with stable scores in [0, 1]
                          or None if stable labels are not available
        """
        if not self.enable_stable_score:
            return None
            
        if obj_idx in self._stable_labels_cache:
            return self._stable_labels_cache[obj_idx]
        
        stable_file = os.path.join(self._stable_labels_path, '{}_stable.npz'.format(str(obj_idx - 1).zfill(3)))
        
        if not os.path.exists(stable_file):
            return None
        
        stable_npz = np.load(stable_file)
        stable_labels = stable_npz['stable'].astype(np.float32)  # (Np, V, A)
        
        # Cache with simple size limit
        if len(self._stable_labels_cache) >= 20:
            oldest = next(iter(self._stable_labels_cache))
            del self._stable_labels_cache[oldest]
        
        self._stable_labels_cache[obj_idx] = stable_labels
        return stable_labels
    
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

        # Rotation along up-axis/Z-axis: Uniform[-30°, +30°]
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        # Random translation: X/Y in [-0.2, 0.2]m, Z in [-0.1, 0.2]m
        translation = np.array([
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.1, 0.2)
        ], dtype=np.float32)
        point_clouds = point_clouds + translation
        for i in range(len(object_poses_list)):
            object_poses_list[i][:, 3] += translation

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
        # sample points
        if self.floor_sampling and self.remove_outlier:
            # Use floor-aware sampling (trans is available when remove_outlier=True)
            idxs = sample_floor_aware(cloud_masked, trans, self.num_points)
        else:
            # Random sampling
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

        if self.use_rgb:
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
        graspness = np.load(self.graspnesspath[index]) 
        scene = self.scenename[index]
        
        if self.use_rgb:
            rgb = np.array(Image.open(self.rgbpath[index]))
        
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
        
        if self.use_rgb:
            rgb_masked = rgb[mask]  # (N, 3) uint8

        # sample points
        if self.floor_sampling and self.remove_outlier:
            # Use floor-aware sampling (trans is available when remove_outlier=True)
            idxs = sample_floor_aware(cloud_masked, trans, self.num_points)
        else:
            # Random sampling
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
        
        grasp_stable_list = []  # List of stable labels per object
        
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
            
            if self.enable_stable_score:
                stable_labels = self._load_stable_labels(obj_idx)
                if stable_labels is not None:
                    grasp_stable_list.append(stable_labels[idxs])  # (Np_sampled, V, A)
                else:
                    grasp_stable_list.append(np.zeros((len(idxs), scores.shape[1], scores.shape[2]), dtype=np.float32))

        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)

        # Shift so all coords are >= 0
        offset = -cloud_sampled.min(axis=0)  # [3,]
        cloud_sampled = cloud_sampled + offset

        if self.use_rgb:
            feats = rgb_sampled.astype(np.float32) / 255.0
        else:
            feats = np.ones_like(cloud_sampled).astype(np.float32)

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats': feats,
                    'cloud_offset': offset.astype(np.float32),
                    'graspness_label': graspness_sampled.astype(np.float32),
                    'objectness_label': objectness_label.astype(np.int64),
                    'object_poses_list': object_poses_list,
                    'grasp_points_list': grasp_points_list,
                    'grasp_widths_list': grasp_widths_list,
                    'grasp_scores_list': grasp_scores_list}
        

        if self.enable_stable_score and len(grasp_stable_list) > 0:
            ret_dict['grasp_stable_list'] = grasp_stable_list
        
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
    
    To ensure that:
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