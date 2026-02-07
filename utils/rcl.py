import torch
import torch.nn.functional as F

def root_cause_localization(
        mse_per_node: torch.Tensor,
        labels: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weights: torch.Tensor,
        alpha_pr: float = 0.85,
        combine_lambda: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-6,
        device: torch.device = None
) -> torch.Tensor:
    """
    Improved Root Cause Localization:
    - Uses the full graph.
    - Calculates scores for all nodes based on MSE and edge weights.
    - Combines direct anomaly score with propagated PageRank score.
    """
    if device is None:
        device = mse_per_node.device

    mse = mse_per_node.to(device)
    # labels are currently unused in calculation but kept for interface compatibility if needed later
    edge_index = edge_index.to(device)
    
    N = mse.shape[0]

    # Handle Edge Weights
    w_scalar = None
    if edge_weights is not None:
        w_scalar = _edge_weights_to_scalar(edge_weights).to(device)
    else:
        # Fallback to uniform weights
        w_scalar = torch.ones(edge_index.size(1), device=device)

    # 1. Calculate Node Strength (incident weight sum)
    node_strength = torch.zeros(N, device=device)
    if edge_index.size(1) > 0:
        for eid in range(edge_index.size(1)):
            s = int(edge_index[0, eid].item())
            t = int(edge_index[1, eid].item())
            w = float(w_scalar[eid].item())
            # Undirected accumulation
            node_strength[s] += w
            node_strength[t] += w

    # 2. Min-Max Normalization Helper
    def minmax_norm(vec: torch.Tensor):
        if vec.numel() == 0:
            return vec
        vmin = vec.min()
        vmax = vec.max()
        if (vmax - vmin) < 1e-12:
            return torch.ones_like(vec) / vec.numel()
        return (vec - vmin) / (vmax - vmin)

    norm_mse = minmax_norm(mse)
    norm_strength = minmax_norm(node_strength)

    # 3. Direct Score: MSE * Strength
    score_direct = norm_mse * (norm_strength + 1e-12)

    # 4. Propagation Score (PageRank-like)
    if N == 1:
        score_pr = norm_mse.clone()
    else:
        # Build Weighted Adjacency Matrix
        W = torch.zeros((N, N), device=device)
        for eid in range(edge_index.size(1)):
            s = int(edge_index[0, eid].item())
            t = int(edge_index[1, eid].item())
            w = float(w_scalar[eid].item())
            W[s, t] += w

        # Row Normalize
        row_sums = W.sum(dim=1)
        P = torch.zeros_like(W)
        nonzero = row_sums > 0
        P[nonzero] = W[nonzero] / row_sums[nonzero].unsqueeze(1)

        # Personalization vector based on MSE
        personalization = norm_mse.clone()
        if personalization.sum() <= 0:
            personalization = torch.ones_like(personalization) / N
        else:
            personalization = personalization / personalization.sum()

        pr = personalization.clone()
        teleport = (1.0 - alpha_pr) * personalization

        # Iteration
        for _ in range(max_iter):
            pr_next = alpha_pr * (P.t() @ pr) + teleport
            if torch.norm(pr_next - pr, p=1) < tol:
                pr = pr_next
                break
            pr = pr_next
        score_pr = minmax_norm(pr)

    # 5. Final Combination
    final_score = combine_lambda * score_direct + (1.0 - combine_lambda) * score_pr
    final_score = final_score.clamp(min=0.0)

    return final_score