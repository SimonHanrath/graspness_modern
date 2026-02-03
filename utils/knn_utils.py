"""
k-Nearest Neighbors utilities for point cloud processing.

Provides efficient kNN implementations using pure PyTorch with:
- Chunked computation for memory efficiency on large point clouds
- Batched processing support
- Flat (N, 3) tensor format for intuitive point cloud handling
"""

import torch

from utils.pointnet.pointnet2_utils import knn_points_torch


# =============================================================================
# Core Implementation - Flat Format (N, 3)
# =============================================================================

@torch.no_grad()
def knn_query(pos: torch.Tensor, k: int, batch: torch.Tensor = None,
              query_pos: torch.Tensor = None, query_batch: torch.Tensor = None) -> torch.Tensor:
    """
    Find k-nearest neighbors for each point using pure PyTorch.
    
    This is a thin wrapper around knn_points_torch that:
    - Converts flat (N, 3) format with batch indices to batched (B, N, D) format
    - For batch=None: reshapes to (1, N, 3), calls knn_points_torch, reshapes back
    - For batch!=None: loops over batches (variable sizes), calls knn_points_torch
      per batch with (1, n_b, 3), then maps local indices to global indices
    - Handles padding when k > number of reference points
    
    Args:
        pos: (N, 3) reference point positions (we search neighbors IN this set)
        k: number of neighbors to find
        batch: (N,) batch indices for each reference point (optional, for batched processing)
        query_pos: (Q, 3) query point positions (optional, defaults to pos for self-kNN)
        query_batch: (Q,) batch indices for query points (optional, defaults to batch)
    
    Returns:
        idx: (Q, k) indices into pos for each query point's k-nearest neighbors
    """
    # Handle self-kNN case
    if query_pos is None:
        query_pos = pos
        query_batch = batch
    elif query_batch is None and batch is not None:
        query_batch = batch
    
    device = pos.device
    N = pos.shape[0]
    Q = query_pos.shape[0]
    
    if batch is None:
        k_actual = min(k, N)
        
        # Fast path for k=1: argmin is faster than topk (common case in the pipeline)
        if k_actual == 1:
            dist = torch.cdist(query_pos.float(), pos.float())  # (Q, N)
            idx = dist.argmin(dim=-1, keepdim=True)  # (Q, 1)
            return idx
        
        p1 = query_pos.unsqueeze(0)  # (1, Q, 3)
        p2 = pos.unsqueeze(0)        # (1, N, 3)
        
        _, idx = knn_points_torch(p1, p2, K=k_actual)  # (1, Q, k_actual)
        idx = idx.squeeze(0)  # (Q, k_actual)
        
        if k > N:
            pad = idx[:, :1].expand(-1, k - N)
            idx = torch.cat([idx, pad], dim=1)
        
        return idx
    else:
        # Batched processing: variable batch sizes require per-batch loop
        # Each batch is processed separately via knn_points_torch, then
        # local indices are mapped to global indices into pos
        unique_batches = torch.unique(batch)
        idx = torch.zeros(Q, k, dtype=torch.long, device=device)
        
        for b in unique_batches:
            ref_mask = (batch == b)
            pos_b = pos[ref_mask]
            ref_indices = torch.where(ref_mask)[0]
            n_b = pos_b.shape[0]
            
            query_mask = (query_batch == b) if query_batch is not None else ref_mask
            query_pos_b = query_pos[query_mask]
            query_indices = torch.where(query_mask)[0]
            q_b = query_pos_b.shape[0]
            
            if n_b == 0 or q_b == 0:
                continue
                
            k_b = min(k, n_b)
            
            # Fast path for k=1
            if k_b == 1:
                dist_b = torch.cdist(query_pos_b.float(), pos_b.float())  # (q_b, n_b)
                local_idx = dist_b.argmin(dim=-1, keepdim=True)  # (q_b, 1)
            else:
                p1_b = query_pos_b.unsqueeze(0)  # (1, q_b, 3)
                p2_b = pos_b.unsqueeze(0)        # (1, n_b, 3)
                
                _, local_idx = knn_points_torch(p1_b, p2_b, K=k_b)  # (1, q_b, k_b)
                local_idx = local_idx.squeeze(0)  # (q_b, k_b)
            
            # Map local indices to global indices
            global_idx = ref_indices[local_idx]
            
            if k_b < k:
                pad_idx = global_idx[:, :1].expand(-1, k - k_b)
                global_idx = torch.cat([global_idx, pad_idx], dim=1)
            
            idx[query_indices] = global_idx
        
        return idx