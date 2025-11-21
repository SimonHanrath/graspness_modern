import torch
from pytorch3d.ops import knn_points

@torch.no_grad()
def knn(ref: torch.Tensor, query: torch.Tensor, k: int = 1,
        lengths_ref: torch.Tensor | None = None,
        lengths_query: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute k-NN in Euclidean space.

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

    ref_bnc   = ref.transpose(1, 2).contiguous()
    query_bqc = query.transpose(1, 2).contiguous()

    # Ensure both tensors have the same dtype for pytorch3d knn_points
    # This is needed when using AMP (Automatic Mixed Precision)
    if ref_bnc.dtype != query_bqc.dtype:
        # Cast both to float32 to ensure compatibility
        ref_bnc = ref_bnc.float()
        query_bqc = query_bqc.float()

    knn = knn_points(query_bqc, ref_bnc, K=min(k, N),
                     lengths1=lengths_query, lengths2=lengths_ref)
    idx_bqk = knn.idx                    # (B, Q, k)

    # Return shape to (B, k, Q) to match the original wrapper
    inds = idx_bqk.permute(0, 2, 1).contiguous()
    return inds.long()