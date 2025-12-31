# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch '''
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from torch.autograd import Function
import torch.nn as nn
import utils.pointnet.pytorch_utils as pt_utils
import sys

from typing import *


class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)



def furthest_point_sample(xyz, npoint) -> torch.Tensor:
    r"""
    Uses iterative furthest point sampling to select a set of npoint features that have the largest
    minimum distance

    Parameters
    ----------
    xyz : torch.Tensor
        (B, N, 3) tensor where N > npoint
    npoint : int32
        number of features in the sampled set

    Returns
    -------
    torch.Tensor
        (B, npoint) tensor containing the set
    """
    if not xyz.is_floating_point():
        xyz = xyz.float()

    B, N, C = xyz.shape
    device = xyz.device
    
    # Initialize with first point (index 0)
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, dtype=xyz.dtype, device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        # Use more efficient squared distance computation
        dist = ((xyz - centroid) ** 2).sum(-1)
        # Update minimum distances
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(-1)
    
    return centroids  # (B, npoint)




def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    features: (B, C, N)
    idx:      (B, npoint) long
    returns:  (B, C, npoint)
    """
    B, C, N = features.shape

    assert idx.dtype == torch.long
    assert idx.device == features.device

    idx_exp = idx.unsqueeze(1).expand(-1, C, -1)      
    out = torch.gather(features, dim=2, index=idx_exp)
    return out


def knn_points_torch(p1: torch.Tensor, p2: torch.Tensor, K: int):
    """
    Pure PyTorch KNN implementation (replacement for pytorch3d.ops.knn_points)
    Memory-efficient version using chunked processing for large point clouds
    
    Parameters
    ----------
    p1 : torch.Tensor
        (B, N, D) query points
    p2 : torch.Tensor
        (B, M, D) reference points
    K : int
        number of nearest neighbors
    
    Returns
    -------
    dists : torch.Tensor
        (B, N, K) squared distances
    idx : torch.Tensor
        (B, N, K) indices of neighbors in p2
    """
    B, N, D = p1.shape
    M = p2.shape[1]
    
    # For small problems, use optimized torch.cdist
    if N * M < 100000:  # Threshold to avoid OOM
        p1_f = p1.float()
        p2_f = p2.float()
        
        # torch.cdist is highly optimized
        dists = torch.cdist(p1_f, p2_f, p=2.0) ** 2  # (B, N, M) - convert back to squared
        
        knn_dists, knn_idx = torch.topk(dists, K, dim=2, largest=False, sorted=True)
        return knn_dists, knn_idx
    
    # For large problems, use chunked processing
    chunk_size = max(1, 50000 // M)  # Process ~50k distances at a time
    all_dists = []
    all_idx = []
    
    p2_f = p2.float()
    
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        p1_chunk = p1[:, i:end_i].float()
        
        dists_chunk = torch.cdist(p1_chunk, p2_f, p=2.0) ** 2
        
        knn_dists, knn_idx = torch.topk(dists_chunk, K, dim=2, largest=False, sorted=True)
        all_dists.append(knn_dists)
        all_idx.append(knn_idx)
    
    return torch.cat(all_dists, dim=1), torch.cat(all_idx, dim=1)


def three_nn(unknown: torch.Tensor, known: torch.Tensor):
    """
    Find the three nearest neighbors of unknown in known Parameters
    unknown: (B, n, 3)
    known:   (B, m, 3)
    returns:
      dist: (B, n, 3)  Euclidean distances to 3-NN in 'known'
      idx:  (B, n, 3)  indices of neighbors in 'known'
    """

    unknown = unknown.to(dtype=torch.float32).contiguous()
    known   = known.to(dtype=torch.float32).contiguous()

    d2, idx = knn_points_torch(unknown, known, K=3)

    # to match pointnet2 API: return euclidean distances, not squared
    dist = torch.sqrt(torch.clamp(d2, min=0.0))
    return dist, idx



def three_interpolate(features: torch.Tensor,
                      idx: torch.Tensor,
                      weight: torch.Tensor) -> torch.Tensor:
    """
    features: (B, C, M)   — source features
    idx:      (B, N, 3)   — neighbor indices into M
    weight:   (B, N, 3)   — weights per neighbor (typically sum to 1)
    returns:  (B, C, N)   — interpolated features at N queries
    """

    assert idx.dtype == torch.long
    assert features.device == idx.device == weight.device

    B, C, M = features.shape
    _, N, K = idx.shape
    assert K == 3

    idx_exp   = idx.unsqueeze(1).expand(-1, C, -1, -1)        # (B, C, N, 3)
    feats_exp = features.unsqueeze(2).expand(-1, -1, N, -1)   # (B, C, N, M)
    gathered  = torch.gather(feats_exp, dim=3, index=idx_exp) # (B, C, N, 3)

    out = (gathered * weight.unsqueeze(1)).sum(dim=3)
    return out


def grouping_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Group features by indices.

    Args:
        features: (B, C, N) float tensor
        idx:      (B, npoint, nsample) long tensor of indices in [0, N)

    Returns:
        grouped:  (B, C, npoint, nsample) float tensor
    """
    assert idx.dtype == torch.long
    assert features.device == idx.device

    B, C, N = features.shape
    B2, npoint, nsample = idx.shape
    assert B == B2

    feats_exp = features.unsqueeze(2).expand(-1, -1, npoint, -1)    # (B, C, npoint, N)
    idx_exp   = idx.unsqueeze(1).expand(-1, C, -1, -1)              # (B, C, npoint, nsample)

    grouped = torch.gather(feats_exp, dim=3, index=idx_exp)         # (B, C, npoint, nsample)
    return grouped



def ball_query(radius: float,
                   nsample: int,
                   xyz: torch.Tensor,
                   new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Same return as original ball_query, but uses KNN then filters by radius.
    """

    d2, idx = knn_points_torch(new_xyz, xyz, K=nsample)

    mask = d2 <= (radius * radius)
   
    B, P, K = idx.shape
    
    any_valid = mask.any(dim=-1, keepdim=True)  # (B,P,1)
    last_valid_pos = torch.where(
        any_valid,
        mask.cumsum(-1).clamp(max=1).sum(-1, keepdim=True) - 1,  # index of last True
        torch.zeros_like(any_valid, dtype=torch.long)
    )  # (B,P,1)
    
    gather_last = torch.gather(idx, -1, last_valid_pos.expand(-1, -1, 1))  # (B,P,1)
    idx = torch.where(mask, idx, gather_last.expand(-1, -1, K))
    return idx

class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind


        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

class QueryAndGroup(nn.Module):
    """
    Groups with a ball query of radius.

    Args:
        radius (float): radius for neighborhood.
        nsample (int): max neighbors per centroid.
        use_xyz (bool): if True, prepend relative xyz to features.
        ret_grouped_xyz (bool): if True, also return grouped_xyz.
        normalize_xyz (bool): if True, divide relative xyz by radius.
        sample_uniformly (bool): if True, ensure indices per region are sampled uniformly from uniques.
        ret_unique_cnt (bool): if True, also return the number of unique neighbors (requires sample_uniformly).
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False,
                 normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        super().__init__()
        self.radius = float(radius)
        self.nsample = int(nsample)
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert self.sample_uniformly

    @torch.no_grad()
    def _uniformize_indices(self, idx: torch.Tensor):
        """
        Make per-(B, npoint) neighbor lists more uniform by sampling from uniques with replacement
        to reach nsample. Also compute unique counts if requested.

        idx: (B, npoint, nsample) long
        returns: (idx_uniform, unique_cnt [optional])
        """
        B, P, K = idx.shape
        device = idx.device
        unique_cnt = None
        if self.ret_unique_cnt:
            unique_cnt = torch.zeros(B, P, device=device, dtype=torch.long)

        for b in range(B):
            for p in range(P):
                uniques = torch.unique(idx[b, p, :], sorted=False)
                num_unique = uniques.numel()
                if self.ret_unique_cnt:
                    unique_cnt[b, p] = num_unique
                if num_unique == 0:
                    continue
                if num_unique < K:
                    extra = torch.randint(0, num_unique, (K - num_unique,), device=device)
                    all_inds = torch.cat([uniques, uniques[extra]], dim=0)
                else:
                    all_inds = uniques[:K]
                idx[b, p, :] = all_inds
        return (idx, unique_cnt) if self.ret_unique_cnt else (idx,)

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        Args:
            xyz: (B, N, 3) coordinates of all points
            new_xyz: (B, npoint, 3) centroid/query points
            features: (B, C, N) descriptors at all points or None

        Returns:
            If features is not None and use_xyz:
                new_features: (B, C+3, npoint, nsample)
            Else:
                new_features: (B, C, npoint, nsample) or (B, 3, npoint, nsample)

            Optionally also returns:
                grouped_xyz: (B, 3, npoint, nsample) if ret_grouped_xyz
                unique_cnt: (B, npoint) long if ret_unique_cnt
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)  # your pure torch / knn-based version

        if self.sample_uniformly:
            out = self._uniformize_indices(idx)
            if self.ret_unique_cnt:
                idx, unique_cnt = out
            else:
                idx = out[0]

        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)                     # (B, 3, npoint, nsample)
        grouped_xyz = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)    # relative coords
        if self.normalize_xyz:
            grouped_xyz = grouped_xyz / self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)             # (B, C, npoint, nsample)
            new_features = (torch.cat([grouped_xyz, grouped_features], dim=1)
                            if self.use_xyz else grouped_features)
        else:
            assert self.use_xyz
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        return ret[0] if len(ret) == 1 else tuple(ret)


@torch.no_grad()
def cylinder_query(
    radius: float,
    hmin: float,
    hmax: float,
    nsample: int,
    xyz: torch.Tensor,         # (B, N, 3)   all points
    new_xyz: torch.Tensor,     # (B, P, 3)   queries/centroids
    rot: torch.Tensor,         # (B, P, 9)   cyl->world rotation matrices (row-major)
) -> torch.Tensor:
    """
    Returns:
        idx: (B, P, nsample) long indices into xyz.
    """
    B, N, _ = xyz.shape
    P = new_xyz.shape[1]
    

    d2_cand, idx_cand = knn_points_torch(
        new_xyz, xyz, K=N
    )  # (B, P, K0), (B, P, K0)

    cand_xyz = torch.gather(
        xyz.unsqueeze(1).expand(B, P, N, 3),      # (B,P,N,3) view
        2,
        idx_cand.unsqueeze(-1).expand(-1, -1, -1, 3)
    )                                             # (B,P,K0,3)
    delta = cand_xyz - new_xyz.unsqueeze(2)       # (B,P,K0,3)

    R_cw = rot.view(B, P, 3, 3)
    R_wc = R_cw.transpose(-1, -2)                 # (B,P,3,3)
    delta_cyl = torch.matmul(delta, R_wc)         # (B,P,K0,3)
    x = delta_cyl[..., 0]
    y = delta_cyl[..., 1]
    z = delta_cyl[..., 2]
    radial = torch.sqrt(torch.clamp(y*y + z*z, min=0.0))

    in_cyl = (radial <= radius) & (x >= hmin) & (x <= hmax)

    score = radial + 1e-3 * torch.abs(x)
    score = score.masked_fill(~in_cyl, float('inf'))  # invalid to +inf

    k = min(nsample, score.size(-1))
    vals, pos = torch.topk(score, k=k, dim=-1, largest=False)   # positions within candidate axis
    chosen = torch.gather(idx_cand, -1, pos)                    # map to original indices (B,P,k)

    if k < nsample:
        pad = nsample - k
        chosen = torch.cat([chosen, chosen[..., -1:].expand(-1, -1, pad)], dim=-1)
        vals   = torch.cat([vals,   vals[...,   -1:].expand(-1, -1, pad)], dim=-1)

    no_valid = torch.isinf(vals[..., 0])
    if no_valid.any():
        fallback = idx_cand[..., 0]                              # nearest overall
        chosen[no_valid] = fallback[no_valid].unsqueeze(-1).expand(-1, nsample)

    return chosen.to(torch.long)


class CylinderQueryAndGroup(nn.Module):
    r"""
    Groups with a cylinder query of radius and height

    Parameters
    ---------
    radius : float32
        Radius of cylinder
    hmin, hmax: float32
        endpoints of cylinder height in x-rotation axis
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, hmin, hmax, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, rotate_xyz=True, sample_uniformly=False, ret_unique_cnt=False):
        super(CylinderQueryAndGroup, self).__init__()
        self.radius, self.nsample, self.hmin, self.hmax, = radius, nsample, hmin, hmax
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.rotate_xyz = rotate_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, rot, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        rot : torch.Tensor
            rotation matrices (B, npoint, 3, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        B, npoint, _ = new_xyz.size()
        idx = cylinder_query(self.radius, self.hmin, self.hmax, self.nsample, xyz, new_xyz, rot.view(B, npoint, 9))

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind


        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        if self.rotate_xyz:
            grouped_xyz_ = grouped_xyz.permute(0, 2, 3, 1).contiguous() # (B, npoint, nsample, 3)
            grouped_xyz_ = torch.matmul(grouped_xyz_, rot)
            grouped_xyz = grouped_xyz_.permute(0, 3, 1, 2).contiguous()


        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)