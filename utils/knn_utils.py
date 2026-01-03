"""
k-Nearest Neighbors utilities for point cloud processing.

Provides efficient kNN implementations using pure PyTorch with:
- Chunked computation for memory efficiency on large point clouds
- Batched processing support
- Flat (N, 3) tensor format for intuitive point cloud handling
"""

import torch


# =============================================================================
# Core Implementation - Flat Format (N, 3)
# =============================================================================

@torch.no_grad()
def knn_query(pos: torch.Tensor, k: int, batch: torch.Tensor = None,
              query_pos: torch.Tensor = None, query_batch: torch.Tensor = None) -> torch.Tensor:
    """
    Find k-nearest neighbors for each point using pure PyTorch.        
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
    
    # Ensure float for distance computation
    pos = pos.float()
    query_pos = query_pos.float()
    
    if batch is None:
        # Single batch - compute pairwise distances
        k_actual = min(k, N)
        
        dist = torch.cdist(query_pos, pos)  # (Q, N)
        _, idx = dist.topk(k_actual, dim=-1, largest=False)  # (Q, k)
        
        # Pad if k > N
        if k > N:
            pad = idx[:, :1].expand(-1, k - N)
            idx = torch.cat([idx, pad], dim=1)
        
        return idx
    else:
        # Batched processing to find neighbors within same batch only
        unique_batches = torch.unique(batch)
        idx = torch.zeros(Q, k, dtype=torch.long, device=device)
        
        for b in unique_batches:
            # Get reference points for this batch
            ref_mask = (batch == b)
            pos_b = pos[ref_mask]
            ref_indices = torch.where(ref_mask)[0]
            n_b = pos_b.shape[0]
            
            # Get query points for this batch
            query_mask = (query_batch == b) if query_batch is not None else ref_mask
            query_pos_b = query_pos[query_mask]
            query_indices = torch.where(query_mask)[0]
            q_b = query_pos_b.shape[0]
            
            if n_b == 0 or q_b == 0:
                continue
                
            k_b = min(k, n_b)
            
            #if q_b * n_b < 50_000_000:
            dist_b = torch.cdist(query_pos_b, pos_b)
            _, local_idx = dist_b.topk(k_b, dim=-1, largest=False)
            """else: TODO: think about removing this
                local_idx = torch.zeros(q_b, k_b, dtype=torch.long, device=device)
                chunk_size = max(1, 25_000_000 // n_b)
                for i in range(0, q_b, chunk_size):
                    end_i = min(i + chunk_size, q_b)
                    dist_chunk = torch.cdist(query_pos_b[i:end_i], pos_b)
                    _, local_idx[i:end_i] = dist_chunk.topk(k_b, dim=-1, largest=False)
            """
            # Map local indices to global indices
            global_idx = ref_indices[local_idx]
            
            # Pad if k_b < k
            if k_b < k:
                pad_idx = global_idx[:, :1].expand(-1, k - k_b)
                global_idx = torch.cat([global_idx, pad_idx], dim=1)
            
            idx[query_indices] = global_idx
        
        return idx