import torch

@torch.no_grad()
def knn(ref: torch.Tensor, query: torch.Tensor, k: int = 1,
        lengths_ref: torch.Tensor | None = None,
        lengths_query: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute k-NN in Euclidean space.
    Memory-efficient implementation with chunked processing.

    Args
    ----
    ref:   (B, C, N)  reference points  (we search neighbors IN this set)
    query: (B, C, Q)  query points      (we search neighbors FOR these)
    k:     int        number of neighbors
    lengths_ref:   optional (B,) valid counts for ref if padded
    lengths_query: optional (B,) valid counts for query if padded

    Returns
    -------
    inds: (B, k, Q) indices into ref for each query
    """
    assert ref.dim() == 3 and query.dim() == 3
    B, C, N = ref.shape
    _, Cq, Q = query.shape
    assert C == Cq

    ref_bnc   = ref.transpose(1, 2).contiguous()    # (B, N, C)
    query_bqc = query.transpose(1, 2).contiguous()  # (B, Q, C)

    # Ensure both tensors are float for distance computation
    ref_bnc = ref_bnc.float()
    query_bqc = query_bqc.float()
    
    K = min(k, N)
    
    # For small problems, use optimized torch.cdist
    if Q * N < 100000:
        dists = torch.cdist(query_bqc, ref_bnc, p=2.0) ** 2  # (B, Q, N) - squared distance
        
        if lengths_ref is not None:
            mask = torch.arange(N, device=ref.device).view(1, 1, N) >= lengths_ref.view(B, 1, 1)
            dists = dists.masked_fill(mask, float('inf'))
        
        _, idx_bqk = torch.topk(dists, K, dim=2, largest=False, sorted=True)
    else:
        # Chunked processing for large problems
        chunk_size = max(1, 50000 // N)
        all_idx = []
        
        for i in range(0, Q, chunk_size):
            end_i = min(i + chunk_size, Q)
            query_chunk = query_bqc[:, i:end_i]
            
            dists = torch.cdist(query_chunk, ref_bnc, p=2.0) ** 2
            
            if lengths_ref is not None:
                mask = torch.arange(N, device=ref.device).view(1, 1, N) >= lengths_ref.view(B, 1, 1)
                dists = dists.masked_fill(mask, float('inf'))
            
            _, idx_chunk = torch.topk(dists, K, dim=2, largest=False, sorted=True)
            all_idx.append(idx_chunk)
        
        idx_bqk = torch.cat(all_idx, dim=1)

    # shape to (B, k, Q)
    inds = idx_bqk.permute(0, 2, 1).contiguous()
    return inds.long()