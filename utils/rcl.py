import torch
import torch.nn.functional as F

def _edge_weights_to_scalar(edge_weights):
    """
    Converts edge weights (possibly multi-head) to scalar.
    Default strategy: Mean over heads.
    """
    if edge_weights is None:
        return None
        
    if edge_weights.dim() == 3 and edge_weights.shape[-1] == 1:
        edge_weights = edge_weights.squeeze(-1)
        
    if edge_weights.dim() == 1:
        return edge_weights
    else:
        return edge_weights.mean(dim=1)

def root_cause_localization(
        mse_per_node: torch.Tensor,
        labels: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weights: torch.Tensor,
        device: torch.device = None,
        alpha_pr: float = 0.85,
        combine_lambda: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-6
) -> tuple:
    """
    Improved Root Cause Localization:
    - Combines direct anomaly score with propagated PageRank score.
    
    Returns:
        tuple: (final_score, score_pr)
        - final_score: Combined score (used for final ranking)
        - score_pr: PageRank score component (used as weight w1 in GAT training)
    """
    if device is None:
        device = mse_per_node.device
    
    mse = mse_per_node.to(device)
    edge_index = edge_index.to(device)
    
    N = mse.shape[0]

    w_scalar = None
    if edge_weights is not None:
        w_scalar = _edge_weights_to_scalar(edge_weights).to(device)
    else:
        w_scalar = torch.ones(edge_index.size(1), device=device)

    def minmax_norm(vec: torch.Tensor):
        if vec.numel() == 0:
            return vec
        vmin = vec.min()
        vmax = vec.max()
        if (vmax - vmin) < 1e-12:
            return torch.ones_like(vec) / (vec.numel() + 1e-12)
        return (vec - vmin) / (vmax - vmin + 1e-12)

    norm_mse = minmax_norm(mse)

    node_strength = torch.zeros(N, device=device)
    if edge_index.size(1) > 0:
        src, dst = edge_index
        node_strength.scatter_add_(0, src, w_scalar)
        node_strength.scatter_add_(0, dst, w_scalar)

    norm_strength = minmax_norm(node_strength)
    score_direct = norm_mse * (norm_strength + 1e-12)

    score_pr = None
    
    if N <= 1:
        score_pr = norm_mse.clone()
    else:
        W = torch.zeros((N, N), device=device)
        src = edge_index[0]
        dst = edge_index[1]
        
        W.index_put_((src, dst), w_scalar)
        
        row_sums = W.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1.0 
        P = W / row_sums

        personalization = norm_mse.clone()
        if personalization.sum() > 0:
            personalization = personalization / personalization.sum()
        else:
            personalization = torch.ones(N, device=device) / N

        pr = personalization.clone()
        
        for _ in range(max_iter):
            pr_next = alpha_pr * torch.mv(P.t(), pr) + (1.0 - alpha_pr) * personalization
            
            if torch.norm(pr_next - pr, p=1) < tol:
                pr = pr_next
                break
            pr = pr_next
            
        score_pr = minmax_norm(pr)

    final_score = combine_lambda * score_direct + (1.0 - combine_lambda) * score_pr
    final_score = final_score.clamp(min=0.0)

    return final_score, score_pr