"""
Utility functions for computing stable score labels.

Stable score indicates how likely a grasp will cause the object to tip/rotate
when lifted. It is computed as the normalized perpendicular distance from the
object's center of gravity (COG) to the gripper plane.

"""

import os
import numpy as np
from tqdm import tqdm

# Try importing mesh libraries (currently used: trimesh)
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def compute_mesh_cog(mesh_path, method='trimesh'):
    """
    Compute center of gravity (volumetric centroid) of a mesh.
    
    Args:
        mesh_path: Path to mesh file (.ply, .obj, etc.)
        method: 'trimesh' (more accurate) or 'vertex' (fallback)
    
    Returns:
        cog: np.ndarray of shape (3,) - center of gravity in object frame
    """
    if method == 'trimesh' and HAS_TRIMESH:
        try:
            mesh = trimesh.load(mesh_path)
            if hasattr(mesh, 'center_mass') and mesh.is_watertight:
                # Use volumetric center of mass for watertight meshes
                return np.array(mesh.center_mass, dtype=np.float32)
            else:
                # Fall back to centroid of vertices
                if hasattr(mesh, 'centroid'):
                    return np.array(mesh.centroid, dtype=np.float32)
                else:
                    # Voxelize and compute centroid of occupied voxels
                    try:
                        voxels = mesh.voxelized(pitch=0.002)  # 2mm voxels
                        return np.array(voxels.points.mean(axis=0), dtype=np.float32)
                    except:
                        pass
        except Exception as e:
            print(f"Warning: trimesh failed for {mesh_path}: {e}")
    
    # Fallback: use vertex centroid
    if HAS_OPEN3D:
        pcd = o3d.io.read_point_cloud(mesh_path)
        points = np.asarray(pcd.points)
        return points.mean(axis=0).astype(np.float32)
    elif HAS_TRIMESH:
        mesh = trimesh.load(mesh_path)
        return np.array(mesh.vertices.mean(axis=0), dtype=np.float32)
    else:
        raise RuntimeError("Need either trimesh or open3d to load meshes for COG computation")


def generate_grasp_views(num_views=300, phi=(np.sqrt(5) - 1) / 2):
    """
    Generate template grasp views on a unit sphere using Fibonacci lattice.
    Same as in utils/loss_utils.py but numpy version.
    
    Returns:
        views: np.ndarray of shape (num_views, 3)
    """
    views = []
    for i in range(num_views):
        zi = (2 * i + 1) / num_views - 1
        xi = np.sqrt(1 - zi ** 2) * np.cos(2 * i * np.pi * phi)
        yi = np.sqrt(1 - zi ** 2) * np.sin(2 * i * np.pi * phi)
        views.append([xi, yi, zi])
    return np.array(views, dtype=np.float32)


def compute_grasp_plane_normal(view, angle=0):
    """
    Compute the gripper plane normal for a given view and in-plane angle.
    
    In GraspNet convention:
    - view: approach direction (gripper approaches along -view direction)
    - angle: in-plane rotation around the approach axis
    - The gripper plane is defined by the approach direction as its normal
    
    Args:
        view: np.ndarray of shape (3,) - unit approach direction
        angle: float - in-plane rotation angle (radians), not used for normal
    
    Returns:
        normal: np.ndarray of shape (3,) - gripper plane normal (= approach dir)
    """
    return -view / (np.linalg.norm(view) + 1e-8)


def compute_stable_score_for_object(grasp_label_path, mesh_path, num_views=300, num_angles=12):
    """
    Compute stable scores for all grasps of a single object.
    
    The stable score measures how likely a grasp will cause tipping.
    It is the normalized perpendicular distance from the object's COG
    to the gripper plane.
    
    Args:
        grasp_label_path: Path to grasp label file (xxx_labels.npz)
        mesh_path: Path to object mesh file
        num_views: Number of template views (default 300)
        num_angles: Number of in-plane angles (default 12)
    
    Returns:
        stable: np.ndarray of shape (Np, V, A) - stable scores in [0, 1]
        cog: np.ndarray of shape (3,) - object center of gravity
    """
    # Load grasp labels
    label = np.load(grasp_label_path)
    grasp_points = label['points']  # (Np, 3) - grasp contact points in object frame
    
    Np = grasp_points.shape[0]
    V = num_views
    A = num_angles
    
    cog = compute_mesh_cog(mesh_path)
    
    views = generate_grasp_views(num_views)  # (V, 3)
    
    stable = np.zeros((Np, V, A), dtype=np.float32)
    
    # For each grasp point
    for p_idx in range(Np):
        grasp_point = grasp_points[p_idx]
        
        # For each view
        for v_idx in range(V):
            view = views[v_idx]
            
            plane_normal = compute_grasp_plane_normal(view)
            
            # Compute perpendicular distance from COG to gripper plane
            # Distance = |n · (cog - point)| where n is unit normal
            cog_to_point = cog - grasp_point
            distance = np.abs(np.dot(plane_normal, cog_to_point))
            
            # Same distance for all angles (stable score is angle-independent)
            stable[p_idx, v_idx, :] = distance
    
    # Normalize per object: divide by max distance
    max_dist = stable.max()
    if max_dist > 1e-8:
        stable = stable / max_dist
    
    return stable, cog


def ensure_stable_labels_exist(dataset_root, num_objects=88, use_simplified=True, verbose=True):
    """
    Check if stable labels exist for all objects, and compute any missing ones.
    
    This function is designed to be called automatically by the dataset when
    enable_stable_score=True. It will only compute labels that are missing,
    making subsequent calls fast.
    
    Args:
        dataset_root: Root directory of GraspNet-1B dataset
        num_objects: Number of objects (default 88 for GraspNet-1B)
        use_simplified: Whether to use simplified grasp labels (recommended)
        verbose: Whether to print progress messages
    
    Returns:
        bool: True if all stable labels are available, False if computation failed
    """
    if not HAS_TRIMESH and not HAS_OPEN3D:
        if verbose:
            print("ERROR: Cannot compute stable scores - need either trimesh or open3d")
            print("Install with: pip install trimesh")
        return False
    
    output_dir = os.path.join(dataset_root, 'stable_labels')
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine grasp label directory
    if use_simplified:
        grasp_label_dir = os.path.join(dataset_root, 'grasp_label_simplified')
    else:
        grasp_label_dir = os.path.join(dataset_root, 'grasp_label')
    
    models_dir = os.path.join(dataset_root, 'models')
    
    # Check which objects need computation
    missing_objects = []
    for obj_idx in range(num_objects):
        obj_id_str = str(obj_idx).zfill(3)
        output_path = os.path.join(output_dir, f'{obj_id_str}_stable.npz')
        if not os.path.exists(output_path):
            missing_objects.append(obj_idx)
    
    if not missing_objects:
        if verbose:
            print(f"All {num_objects} stable label files already exist in {output_dir}")
        return True
    
    if verbose:
        print(f"Computing stable scores for {len(missing_objects)} missing objects...")
        print(f"  Grasp labels: {grasp_label_dir}")
        print(f"  Models: {models_dir}")
        print(f"  Output: {output_dir}")
    
    cog_records = {}
    iterator = tqdm(missing_objects, desc="Computing stable labels") if verbose else missing_objects
    
    for obj_idx in iterator:
        obj_id_str = str(obj_idx).zfill(3)
        
        # Paths
        grasp_label_path = os.path.join(grasp_label_dir, f'{obj_id_str}_labels.npz')
        mesh_path = os.path.join(models_dir, obj_id_str, 'nontextured.ply')
        output_path = os.path.join(output_dir, f'{obj_id_str}_stable.npz')
        
        # Check if files exist
        if not os.path.exists(grasp_label_path):
            if verbose:
                print(f"Warning: Grasp labels not found for object {obj_idx}: {grasp_label_path}")
            continue
        if not os.path.exists(mesh_path):
            if verbose:
                print(f"Warning: Mesh not found for object {obj_idx}: {mesh_path}")
            continue
        
        try:
            stable, cog = compute_stable_score_for_object(grasp_label_path, mesh_path)
            
            # Save results
            np.savez_compressed(output_path, stable=stable, cog=cog)
            cog_records[obj_idx] = cog
            
        except Exception as e:
            if verbose:
                print(f"Error processing object {obj_idx}: {e}")
            continue
    
    # Update summary file
    summary_path = os.path.join(output_dir, 'summary.npz')
    
    # Load existing summary if present and merge
    existing_cogs = {}
    if os.path.exists(summary_path):
        try:
            summary = np.load(summary_path, allow_pickle=True)
            existing_ids = summary.get('object_ids', [])
            existing_cog_array = summary.get('cogs', np.array([]))
            for i, oid in enumerate(existing_ids):
                if i < len(existing_cog_array):
                    existing_cogs[int(oid)] = existing_cog_array[i]
        except:
            pass
    
    # Merge with new COGs
    existing_cogs.update(cog_records)
    
    if existing_cogs:
        sorted_ids = sorted(existing_cogs.keys())
        np.savez(summary_path,
                 num_objects=len(sorted_ids),
                 object_ids=sorted_ids,
                 cogs=np.stack([existing_cogs[oid] for oid in sorted_ids]))
    
    if verbose:
        print(f"Done! Stable labels ready for {num_objects - len(missing_objects) + len(cog_records)} objects.")
    
    return True


def check_stable_labels_available(dataset_root, num_objects=88):
    """
    Quick check if all stable labels exist without computing.
    
    Returns:
        tuple: (all_exist: bool, missing_count: int)
    """
    output_dir = os.path.join(dataset_root, 'stable_labels')
    missing = 0
    for obj_idx in range(num_objects):
        obj_id_str = str(obj_idx).zfill(3)
        output_path = os.path.join(output_dir, f'{obj_id_str}_stable.npz')
        if not os.path.exists(output_path):
            missing += 1
    return missing == 0, missing
