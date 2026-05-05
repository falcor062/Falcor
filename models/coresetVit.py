import logging
from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import PatchEmbed, Mlp, DropPath
from torch.nn import Module

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

class BatchedTokenSelector(nn.Module):
    def __init__(self, K, α=0.5, β=1.0, learnable_ab=True):
        super().__init__()
        self.K = K
        self.α = nn.Parameter(torch.tensor(α)) if learnable_ab else α
        self.β = nn.Parameter(torch.tensor(β)) if learnable_ab else β

    def soft_topk(self, scores, K, τ=0.5):
        """Gumbel–top-K relaxation (Xie & Ermon, 2020)."""
        g = -torch.log(-torch.log(torch.rand_like(scores)))  # Gumbel noise
        y = (scores + g) / τ  # perturb & temper
        # Sinkhorn-style soft ranking → K-hot weights in (0,1)
        logits = y - torch.logsumexp(y, dim=-1, keepdim=True)  # subtract log-sum-exp
        weights = torch.sigmoid(logits * 10)  # sharpen
        # Re-scale so Σ weights ≈ K
        weights = K * weights / weights.sum(dim=-1, keepdim=True)
        return weights.clamp(0, 1)

    def forward(self, tokens, A, B, train=True):
        """
        tokens : [B, N, D]
        A      : [B, H, N, N]
        B      : [B, N]  or  [B, H, N]  (see ▼)
        """
        B_  = B if B.ndim == 3 else B.unsqueeze(1)        # → [B,H,N]

        # --- 1.  quality ----------------------------------------------
        q = self.α * B_.mean(1) + (1 - self.α) * A.sum(-2).mean(1)   # (B,N)

        # --- 2.  redundancy (variation-of-information) ----------------
        p = torch.softmax(A, dim=-1)                      # (B,H,N,N)
        H = -(p * p.log()).sum(-1, keepdim=True)          # (B,H,N,1)
        CE = -(p @ p.log().transpose(-1,-2))              # (B,H,N,N)
        MI = 0.5 * (H + H.transpose(-1,-2) - CE - CE.transpose(-1,-2))
        r = MI.mean(-1).mean(1)                           # (B,N)

        # --- 3.  combined score & mask -------------------------------
        score = q - self.β * r                            # (B,N)

        if self.training:
            w = self.soft_topk(score, self.K)                  # (B,N)
            return tokens * w.unsqueeze(-1)               # masked tokens
        else:                                             # inference
            idx = score.topk(self.K, dim=-1).indices      # (B,K)
            gathered = torch.gather(tokens, 1,
                                     idx.unsqueeze(-1).expand(-1,-1,tokens.size(-1)))
            return gathered

class TokenSelector(nn.Module):

    def __init__(self, K, α=0.5, β=1.0):
        super().__init__()
        self.K = K
        self.α = nn.Parameter(torch.tensor(α))  # optional learnable
        self.β = nn.Parameter(torch.tensor(β))

    def soft_topk(self, scores, K, τ=0.5):
        """Gumbel–top-K relaxation (Xie & Ermon, 2020)."""
        g = -torch.log(-torch.log(torch.rand_like(scores)))  # Gumbel noise
        y = (scores + g) / τ  # perturb & temper
        # Sinkhorn-style soft ranking → K-hot weights in (0,1)
        logits = y - torch.logsumexp(y, dim=-1, keepdim=True)  # subtract log-sum-exp
        weights = torch.sigmoid(logits * 10)  # sharpen
        # Re-scale so Σ weights ≈ K
        weights = K * weights / weights.sum(dim=-1, keepdim=True)
        return weights.clamp(0, 1)

    def forward(self, tokens, A, B, train=True):
        """
        A is the attention matrix NxN weights
        B is the token-cls weight
        """
        # 1. quality
        q = self.α * B + (1 - self.α) * A.sum(dim=0)  # (N,)

        # 2. redundancy
        p = torch.softmax(A, dim=-1)  # rows → probs
        H = -(p * p.log()).sum(-1, keepdim=True)
        X = -(p @ p.log().T)
        MI = (H + H.T - X - X.T) / 2  # Variation-of-Information
        r = MI.mean(dim=1)  # (N,)

        # 3. combined score
        score = q - self.β * r  # (N,)

        if train:
            w = self.soft_topk(score, self.K)  # (N,) ∈ (0,1)
            tokens = tokens * w.unsqueeze(-1)  # mask in place
            return tokens  # classification loss follows
        else:
            idx = score.topk(self.K).indices
            return tokens[idx], idx  # hard subset


def quality_distance(Q, atten):
    """
    Build a pairwise distance matrix emphasizing quality and attention.
    Q:      [B,N] quality scores per token
    atten:  [B,H,N,N] raw attention maps
    Returns: [B,N,N] quality-aware distance
    """
    B, N = Q.shape

    # --- (1) absolute difference of Q ---
    q_diff = (Q.unsqueeze(-1) - Q.unsqueeze(-2)).abs()  # [B,N,N]

    # --- (2) attention similarity converted to distance ---
    #A = atten.mean(dim=1)  # average heads, [B,N,N]
    A = atten
    A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)  # row normalized
    attn_dist = 1.0 - A                           # higher when less connected

    # --- (3) blend ---
    D_qual = 0.5 * q_diff + 0.5 * attn_dist
    return D_qual


import torch

def attn_stats(atten: torch.Tensor, eps: float = 1e-6) -> dict:
    """
    Batched attention stats for [B, N, N] (no padding).
    Returns per-sample scalars [B]:
      - entropy : mean row entropy
      - margin  : mean (top1 - top2) per row
      - top1    : mean rowwise top-1 prob
      - inc_std : std over columns of incoming attention (column-sum dispersion)
    """
    assert atten.dim() == 3, "atten must be [B, N, N]"
    B, N, _ = atten.shape

    # Row-normalize (query-wise)
    A = atten / (atten.sum(dim=-1, keepdim=True) + eps)         # [B, N, N]
    P = A.clamp_min(eps)

    # Rowwise entropy, top1, margin
    row_entropy = -(P * P.log()).sum(dim=-1)                     # [B, N]
    entropy = row_entropy.mean(dim=1)                            # [B]

    top2 = P.topk(2, dim=-1).values                              # [B, N, 2]
    row_top1   = top2[..., 0]                                    # [B, N]
    row_margin = top2[..., 0] - top2[..., 1]                     # [B, N]
    top1  = row_top1.mean(dim=1)                                 # [B]
    margin= row_margin.mean(dim=1)                               # [B]

    # Incoming centrality dispersion (how unevenly tokens are used as keys)
    col_sums = A.sum(dim=-2)                                     # [B, N]
    inc_std  = col_sums.std(dim=1, unbiased=False)               # [B]

    return {"entropy": entropy, "margin": margin, "top1": top1, "inc_std": inc_std}



import torch.nn.functional as F
import torch

import torch
import torch.nn.functional as F

def quality_modulated_distance(
    D_cos: torch.Tensor,   # [B,N,N], base cosine distance = 1 - cos_sim
    Q: torch.Tensor,       # [B,N],   token quality (any scale)
    atten: torch.Tensor | None = None,  # [B,N,N] optional attention map (row-normalized inside)
    alpha: torch.Tensor | float = 0.5,  # modulation strength in [0,1] (can be per-batch [B,1,1])
    q_sharp: float = 1.0,               # sharpen Q influence (>=1)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Return D_star = D_cos * (1 + alpha * M), where M is built from Q (and optionally atten).
    - Quality *modulates* geometry; quality is not used as a distance term.
    - No padding; all tokens valid.
    """
    B, N, _ = D_cos.shape
    device, dtype = D_cos.device, D_cos.dtype

    # ---- 1) Per-token gate from Q (normalized & sharpened) ----
    # Use softmax to get a stable distribution in each image
    q_prob = F.softmax(Q, dim=-1).clamp_min(eps)              # [B,N]
    q_gate = q_prob.pow(q_sharp)                              # [B,N]
    q_gate = q_gate / (q_gate.sum(dim=-1, keepdim=True) + eps)# [B,N] (still sums to 1)

    # Pairwise gate from two tokens' gates (geometric mean is symmetric & bounded)
    G_pair = torch.sqrt(q_gate.unsqueeze(-1) * q_gate.unsqueeze(-2))  # [B,N,N]

    # ---- 2) Optional attention-based redundancy boost ----
    # If two tokens attend similarly/strongly to each other, inflate distance more.
    if atten is not None:
        A = atten / (atten.sum(dim=-1, keepdim=True) + eps)   # row-normalize [B,N,N]
        A_sym = 0.5 * (A + A.transpose(-1, -2))               # symmetry
        # Use A_sym as a redundancy *similarity*. Higher -> more inflation.
        R = A_sym
    else:
        # If no attention is given, modulation is purely quality-driven
        R = torch.ones_like(D_cos)

    # ---- 3) Final modulation matrix in [0, 1] (bounded & smooth) ----
    # Normalize R per image so its scale is comparable
    Rn = R / (R.amax(dim=(1,2), keepdim=True).clamp_min(1.0))
    M = (G_pair * Rn).clamp(0.0, 1.0)                          # [B,N,N]

    # ---- 4) Blend with alpha (could be scalar, [B,1,1], or learnable) ----
    if not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, device=device, dtype=dtype)
    if alpha.dim() == 0:
        alpha = alpha.view(1,1,1).expand(B,1,1)                # [B,1,1]
    elif alpha.dim() == 1:  # [B] → [B,1,1]
        alpha = alpha.view(B,1,1)

    D_star = D_cos * (1.0 + alpha * M)                         # [B,N,N]
    return D_star

def q_stats(Q: torch.Tensor, eps: float = 1e-6) -> dict:
    """
    Batched quality stats for [B, N].
    Assumes all tokens are valid (no padding).

    Returns dict of [B]-shaped tensors:
      - mean   : average quality score
      - std    : standard deviation of quality scores
      - max    : maximum quality score
      - entropy: normalized entropy over Q (distribution sharpness)
    """
    assert Q.dim() == 2, "Expected Q of shape [B, N]"
    B, N = Q.shape

    # Normalize to a distribution for entropy
    probs = Q.clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True)  # [B, N]

    # Stats
    mean   = Q.mean(dim=1)                     # [B]
    std    = Q.std(dim=1, unbiased=False)      # [B]
    maxval = Q.max(dim=1).values               # [B]

    entropy = -(probs * probs.log()).sum(dim=-1) / torch.log(torch.tensor(N, device=Q.device, dtype=Q.dtype))

    return {
        "mean": mean,       # [B]
        "std": std,         # [B]
        "max": maxval,      # [B]
        "entropy": entropy  # [B], normalized to [0,1]
    }


import torch

def cls_stats(
    x: torch.Tensor,            # [B, C] logits or probs
    input_is_logits: bool = True,
    eps: float = 1e-6,
) -> dict:
    """
    Batched CLS confidence stats.
    Returns dict of [B]-shaped tensors:
      - margin  : top1 - top2 (on probs)
      - entropy : entropy of predicted class distribution
      - top1    : top1 probability
      - neglogp : -log p(top1)  (optional alternative to entropy)
    """
    assert x.dim() == 2, "Expected [B, C]"

    # Convert to probabilities
    if input_is_logits:
        probs = F.softmax(x, dim=-1)
    else:
        probs = x / (x.sum(dim=-1, keepdim=True).clamp_min(eps))

    probs = probs.clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize after clamp

    # Top-1 / Top-2
    top2 = probs.topk(2, dim=-1).values            # [B, 2]
    top1_prob = top2[:, 0]                         # [B]
    margin    = top2[:, 0] - top2[:, 1]           # [B]

    # Entropy and negative log p(top1)
    entropy = -(probs * probs.log()).sum(dim=-1)  # [B]
    neglogp = -(top1_prob.clamp_min(eps)).log()   # [B]

    return {
        "margin":  margin,     # [B]
        "entropy": entropy,    # [B]
        "top1":    top1_prob,  # [B]
        "neglogp": neglogp,    # [B]
    }


import torch


import torch

def geometry_stats_from_matrix(
    D_cos: torch.Tensor,   # [B, N, N] cosine distance (1 - cos sim)
    Q: torch.Tensor,       # [B, N]    quality scores
    M: int = 8,
) -> dict:
    """
    Compute per-batch geometry diagnostics from a precomputed cosine distance matrix.
    No masking (all tokens valid).

    Returns a dict of [B]-shaped tensors:
      - mean_dist   : mean of all pairwise distances (upper triangle)
      - std_dist    : std  of all pairwise distances (upper triangle)
      - min_dist    : min  of all pairwise distances (upper triangle)
      - max_dist    : max  of all pairwise distances (upper triangle)
      - topM_redund : mean pairwise distance among the top-M tokens by Q
                       (higher => less redundant / more diverse)
      - M_eff       : effective M actually used per batch (scalar int, same across batch)
    """
    assert D_cos.dim() == 3 and D_cos.size(1) == D_cos.size(2), "D_cos must be [B,N,N]"
    assert Q.dim() == 2 and Q.size(0) == D_cos.size(0), "Q must be [B,N] matching D_cos"

    B, N, _ = D_cos.shape
    device = D_cos.device

    # ---- Global pairwise stats (use upper triangle, exclude diagonal) ----
    iu = torch.triu_indices(N, N, offset=1, device=device)     # [2, N*(N-1)/2]
    dists = D_cos[:, iu[0], iu[1]]                              # [B, P]
    mean_dist = dists.mean(dim=1)                               # [B]
    std_dist  = dists.std(dim=1, unbiased=False)                # [B]
    min_dist  = dists.min(dim=1).values                         # [B]
    max_dist  = dists.max(dim=1).values                         # [B]

    # ---- Redundancy among top-M by Q ----
    M_eff = int(min(M, N))
    # Top indices per batch
    _, top_idx = Q.topk(M_eff, dim=1)                           # [B, M_eff]
    top_idx = top_idx.long()

    # Advanced indexing to get per-batch submatrices: sub[b] = D_cos[b][top_idx[b]][:, top_idx[b]]
    b = torch.arange(B, device=device)
    sub = D_cos[b[:, None, None],                               # [B,1,1]
                top_idx[:, :, None],                            # [B,M,1]
                top_idx[:, None, :]]                            # [B,1,M]  -> [B,M,M]

    # Mean off-diagonal distance within each top-M set
    iuM = torch.triu_indices(M_eff, M_eff, offset=1, device=device)
    sub_off = sub[:, iuM[0], iuM[1]]                            # [B, M_eff*(M_eff-1)/2]
    topM_redund = sub_off.mean(dim=1) if M_eff > 1 else torch.zeros(B, device=device, dtype=D_cos.dtype)

    return {
        "mean_dist":   mean_dist,     # [B]
        "std_dist":    std_dist,      # [B]
        "min_dist":    min_dist,      # [B]
        "max_dist":    max_dist,      # [B]
        "topM_redund": topM_redund,   # [B], higher => less redundant (more spread) among top-M
        "M_eff":       M_eff,         # int
    }


def geometry_stats(
    features: torch.Tensor,           # [B, N, D]
    Q: torch.Tensor,                  # [B, N] (for picking top-M)
    token_mask: torch.Tensor | None = None,  # [B, N], 1=valid, 0=pad
    M: int = 8,                       # top-M by Q to assess redundancy
    metric: str = "cosine",           # "cosine" | "euclidean"
    eps: float = 1e-6,
) -> dict:
    """
    Returns batched geometry stats:
      - feat_spread       : [B]  mean distance of tokens to per-sample mean feature
      - topQ_redundancy   : [B]  mean pairwise similarity among top-M Q tokens
      - pair_mean (opt)   : [B]  mean pairwise similarity across all valid tokens (cheap cosine-only)
      - pair_std  (opt)   : [B]  std of pairwise similarity across all valid tokens (cheap cosine-only)

    Notes:
      * Uses masking for variable N.
      * For cosine, features are L2-normalized.
      * Redundancy uses pairwise similarity among top-M only (O(M^2)).
    """
    assert features.dim() == 3 and Q.dim() == 2
    B, N, D = features.shape
    device, dtype = features.device, features.dtype

    if token_mask is None:
        token_mask = torch.ones(B, N, device=device, dtype=dtype)
    valid_counts = token_mask.sum(dim=1).clamp_min(1.0)  # [B]

    # -------- Normalize (for cosine) --------
    if metric == "cosine":
        X = F.normalize(features, dim=-1, eps=eps)  # [B,N,D]
    elif metric == "euclidean":
        X = features
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    # -------- Per-sample mean feature (masked) --------
    mu = (X * token_mask.unsqueeze(-1)).sum(dim=1) / valid_counts.unsqueeze(-1)  # [B,D]

    # -------- Feature spread (mean distance to mean) --------
    if metric == "cosine":
        # cosine “distance” to mean dir
        mu_n = F.normalize(mu, dim=-1, eps=eps)                                   # [B,D]
        cos_sim = (X * mu_n.unsqueeze(1)).sum(dim=-1)                              # [B,N]
        dist_to_mu = (1.0 - cos_sim).clamp_min(0)                                  # [B,N]
    else:
        dist_to_mu = (X - mu.unsqueeze(1)).norm(dim=-1)                            # [B,N]

    feat_spread = (dist_to_mu * token_mask).sum(dim=1) / valid_counts              # [B]

    # -------- Top-M redundancy among high-Q tokens --------
    M_eff = int(min(M, int(valid_counts.max().item())))
    # Mask Q for padding before topk
    Q_masked = Q + (1.0 - token_mask) * (-1e9)
    top_idx = Q_masked.topk(M_eff, dim=1).indices                                  # [B,M_eff]
    top_feat = X.gather(
        1, top_idx.unsqueeze(-1).expand(-1, -1, D)
    )                                                                               # [B,M_eff,D]

    # pairwise similarity a




import torch, torch.nn as nn, torch.nn.functional as F

class QualityDiversityBalancer(nn.Module):
    """
    Outputs lam_q in [0,1] (weight on quality distance).
    Input x is a per-image feature vector of summary stats.
    """
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden//2), nn.ReLU(inplace=True),
            nn.Linear(hidden//2, 1)  # -> lam_q logit
        )

    def forward(self, x):  # x: [B, in_dim]
        lam_q = torch.sigmoid(self.net(x))  # [B, 1]
        return lam_q



class CoresetSelectionLayer(Module):
    def __init__(self, selection_count, dim):
        super().__init__()
        #self.gamma = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.0))  # for λ
        self.beta = nn.Parameter(torch.tensor(0.0))  # for
        self.gamma1 = nn.Parameter(torch.tensor(0.0))
        self.q_exp = nn.Parameter(torch.tensor(1.0))
        self.sim_exp = nn.Parameter(torch.tensor(1.0))
        #self.gamma = nn.Parameter(torch.tensor(0.0))
        self.gamma.requires_grad = True
        self.selection_count = selection_count
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.tau.requires_grad = True
        self.gamma_predictor_s1 = nn.Linear(dim, 1)
        init_n = 14 * 14
        #self.gamma_predictor_s2 = nn.Linear(init_n * 4, 1)
        self.gamma_token = nn.Parameter(torch.randn(1, 1, dim))
        self.gamma_mha = nn.MultiheadAttention(embed_dim=dim,num_heads=4, batch_first=True)
        self.gamma_max = nn.Parameter(torch.tensor(1.0))  # Max diversity strength
        self.gamma_decay_power = 1.0  # Controls how fast γ decays with K
        self.gamma_head = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        # in_dim = 7
        # self.balancer = QualityDiversityBalancer(in_dim=in_dim, hidden=32)

    def angular_dist_with_norm(self, candidate_point, quality, input_points):
        """
        a, b are normalized face embeddings.
        Args:
            norm:
            a:
            b:

        Returns:

        """
        self.gamma.requires_grad = True
        input_points = torch.nn.functional.normalize(input_points, p=2.0, dim = -1)
        candidate_point = torch.nn.functional.normalize(candidate_point, p=2.0, dim = -1)

        inner_product = torch.bmm(input_points, candidate_point.transpose(1,2))
        dist = torch.abs((1 - inner_product) / 2)
        candidate_importance = quality.clone()

        candidate_importance = candidate_importance.reshape_as(dist)
        #dist = torch.pow(candidate_importance, self.gamma.relu()) * dist

        min_value = 1e-8
        dist = torch.clamp(dist, min=min_value)
        candidate_importance = torch.clamp(candidate_importance, min=min_value)

        # Compute exponents

        gamma = self.gamma.sigmoid()
        # gamma = 10000000.0
        # exponent1 = 1.0 / (gamma + 1.0)
        # exponent2 = gamma / (gamma + 1.0)
        # gamma = gamma.unsqueeze(-1)
        # gamma = gamma.repeat(1,dist.shape[1], 1)
        quality_importance = gamma
        dist_importance = 1.0 - quality_importance

        quality_dist = torch.pow(dist, dist_importance) * torch.pow(candidate_importance, quality_importance)

        #print(f"dist min: {dist.min()}, dist max: {dist.max()}")
        #print(f"candidate_importance min: {candidate_importance.min()}, candidate_importance max: {candidate_importance.max()}")
        # Compute quality_dist using logarithms for numerical stability
        #quality_dist = dist_importance * torch.log(dist) + quality_importance * torch.log(candidate_importance)
        #quality_dist = torch.exp(log_quality_dist)
        #quality_dist = log_quality_dist

        #quality_dist = log_quality_dist
        #quality_dist = dist

        #quality_dist = torch.pow(candidate_importance, self.gamma.relu()) * dist
        # epsilon = 1e-8
        # dist = dist + epsilon
        # candidate_importance = candidate_importance + epsilon
        # quality_dist = torch.pow(dist.clone(), 1.0 / (self.gamma + 1.0)) * torch.pow(candidate_importance.clone(),
        #                                                                              self.gamma / (self.gamma + 1.0))

#        quality_dist = torch.pow(dist, 1.0 / (self.gamma + 1.0)) * torch.pow(candidate_importance, self.gamma / ( self.gamma + 1.0))
        #quality_dist = candidate_importance * self.gamma.relu() * dist

        # grad_hook_quality_dist = quality_dist.register_hook(
        #     lambda grad: print("quality_dist is {0}".format(grad.data.norm(2))))

        #dist = dist.clip(0,1)
        return quality_dist

    def forward(self, x, quality):
        """
        Forward pass to select features using soft selection with softmax probabilities.

        Args:
            x (torch.Tensor): Input features of shape (batch_size, num_features, feature_dim).
            quality (torch.Tensor): Quality scores for each feature of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Core template of selected features.
        """
        # Determine scaling factor and temperature
        if self.training:
            scaler = 1e6
            #tau = self.tau
            tau = 1e-10  # Near-zero temperature for hard selection
        else:
            scaler = 1e6
            tau = 1e-10  # Near-zero temperature for hard selection

        gamma = self.gamma
        # b = x.shape[0]
        # gamma_tokens = self.gamma_token.expand(b, -1, -1)
        # x_concat_token = torch.cat([gamma_tokens, x], dim=1)
        # out = self.gamma_mha(x_concat_token, x_concat_token,x_concat_token)[0]
        # out = out[:,0,:]
        # gamma = self.gamma_predictor_s1(out).sigmoid()

        # gamma_low_dim_features = self.gamma_predictor_s1(x)
        # gamma = self.gamma_predictor_s2(gamma_low_dim_features.reshape(x.shape[0],-1,1).squeeze(-1)).sigmoid()
        # Rescale quality scores
        scaled_quality = quality * scaler

        # Step 1: Compute softmax probabilities
        first_selection_probs = torch.nn.functional.softmax(scaled_quality / tau, dim=1)
        broadcast_first_selection_probs = first_selection_probs.expand_as(x)
        weighted_features = broadcast_first_selection_probs * x
        core_template = weighted_features.sum(dim=1).unsqueeze(1)
        # Compute the core template as a weighted sum

        # Exclude the selected feature by zeroing out its probability
        mask = (first_selection_probs < 1e-8).float()
        quality = quality * mask

        # Compute distances from the selected core template to all features
        dist_core_to_template = self.angular_dist_with_norm(core_template, quality, x)

        # Iteratively select features
        for _ in range(self.selection_count - 1):
            # Compute softmax probabilities based on distance
            scaled_dist = dist_core_to_template * scaler
            next_selection_probs = torch.nn.functional.softmax(scaled_dist / tau, dim=1)

            broadcast_first_selection_probs = next_selection_probs.expand_as(x)
            weighted_features = broadcast_first_selection_probs * x
            new_core_item = weighted_features.sum(dim=1).unsqueeze(1)

            # Exclude the selected feature
            mask = (next_selection_probs < 1e-8).float()
            quality = quality * mask

            # Update distances
            dist_new_to_template = self.angular_dist_with_norm(new_core_item, quality, x)
            dist_core_to_template = torch.min(dist_core_to_template, dist_new_to_template)

            # Append the new core item
            core_template = torch.cat([core_template, new_core_item], dim=1)

        return core_template

    def variation_of_information_distance_matrix(self, F, Q, gamma, eps=1e-8):
        """
        Compute the variation of information (VI) distance matrix from features F.

        Args:
            F (Tensor): Input logits/features of shape (B, N, D),
                        typically representing attention logits over tokens.
            Q (Tensor): Quality scores of shape (B, N).
            gamma (Tensor): Balancing parameter between quality and diversity.
            eps (float): Small value for numerical stability.

        Returns:
            VI_dist (Tensor): VI-based distance matrix of shape (B, N, N),
                              combined with quality Q according to gamma.
        """
        # Normalize F into probability distributions across the last dimension (D).
        P = torch.softmax(F, dim=-1).clamp(min=eps)  # shape: (B, N, D)

        # Compute entropy H(p) for each token distribution.
        H = -(P * P.log()).sum(dim=-1, keepdim=True)  # shape: (B, N, 1)

        # Compute cross-entropy between pairs.
        CE = -(P @ P.transpose(-1, -2).log().clamp(min=eps))  # shape: (B, N, N)

        # Compute mutual information matrix MI
        MI = (H + H.transpose(-1, -2) - CE - CE.transpose(-1, -2)) / 2.0  # (B, N, N)

        # Compute VI distance from MI: higher MI -> lower VI
        VI_distance = H + H.transpose(-1, -2) - 2 * MI  # (B, N, N)
        VI_distance = VI_distance.clamp(min=eps)

        # Integrate quality scores Q and VI_distance using gamma.
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # shape: (B, 1, 1)
        quality_importance = gamma.repeat(1, VI_distance.shape[1], 1)
        dist_importance = 1.0 - quality_importance

        # Normalize Q for numerical stability and ensure positivity.
        #Q_normalized = Q.unsqueeze(1).repeat(1, VI_distance.shape[1], 1).clamp(min=eps)

        Q_normalized = Q.repeat(1, 1, VI_distance.shape[1]).clamp(min=eps)

        # Combine distances and quality scores.
        combined_log_distance = (dist_importance * VI_distance.log() +
                                 quality_importance * Q_normalized.log())

        VI_dist = combined_log_distance.exp().transpose(-1, -2)  # (B, N, N)

        return VI_dist

    def cosine_distance_matrix(self, F ):
        """
        Compute the cosine distance matrix A from features F.

        Args:
            F (Tensor): Input features of shape (B, N, D),
                        where B is the batch size, N is the number of features,
                        and D is the dimension of each feature.
            eps (float): A small value added for numerical stability.

        Returns:
            A (Tensor): Cosine distance matrix of shape (B, N, N).
                        Each element A[b, i, j] = 1 - cosine_similarity(F[b, i], F[b, j]).
        """
        # Normalize the feature vectors along the last dimension.
        self.gamma.requires_grad = True
        F_norm = F / (F.norm(dim=-1, keepdim=True))  # shape: (B, N, D)

        # Compute cosine similarity via batched matrix multiplication.
        # We multiply F_norm with its transpose along the last two dimensions.
        cosine_sim = torch.matmul(F_norm, F_norm.transpose(-1, -2))  # shape: (B, N, N)
        cosine_sim = (cosine_sim + 1.0) / 2.0
        # Convert cosine similarity to cosine distance.
        cosine_distance = 1 - cosine_sim
        cosine_distance = torch.clamp(cosine_distance, min=1e-8, max=1.0)

        # gamma = self.gamma.unsqueeze(-1).unsqueeze(-1)
        # gamma = gamma.repeat(1,cosine_distance.shape[1], 1)
        # quality_importance = gamma
        # dist_importance = 1.0 - quality_importance

        #gil softmax the distance matrix to normalize relative the attention values which are a softmax product
        #cosine_distance = cosine_distance.softmax(dim=1)
        # gamma = self.gamma.sigmoid()
        # cosine_distance = (1 - gamma) * cosine_distance + gamma * Q
        #cosine_distance =  dist_importance * torch.log(cosine_distance) + quality_importance * torch.log(Q)
        #cosine_distance = torch.exp(cosine_distance)
        #quality_dist = torch.pow(cosine_distance, 1 - gamma) * torch.pow(Q, gamma)
        quality_dist = cosine_distance.transpose(-1, -2)
        return quality_dist

    def compute_entropy(self, probs):
        """Compute entropy given probabilities."""
        return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

    def compute_features(self, quality, attention=None):
        # Attention statistics
        # mean_a = attention.mean(dim=(1, 2))  # Shape: [B]
        # var_a = attention.var(dim=(1, 2))  # Shape: [B]
        # attention_probs = attention / attention.sum(dim=(1, 2), keepdim=True)  # Shape: [B, N, N]
        # entropy_a = compute_entropy(
        #     attention_probs.view(attention_probs.shape[0], -1))  # Flatten [N, N] to [N*N], Shape: [B]

        # Quality statistics
        quality = quality.squeeze(-1)  # Shape: [B, N]
        # mean_q = quality.mean(dim=1)  # Shape: [B]
        # var_q = quality.var(dim=1)  # Shape: [B]
        quality_probs = quality / quality.sum(dim=1, keepdim=True)  # Shape: [B, N]
        entropy_q = self.compute_entropy(quality_probs)  # Shape: [B]

        # Combine features
        # features = torch.stack([mean_a, var_a, entropy_a, mean_q, var_q, entropy_q], dim=1)  # Shape: [B, 8]
        # features = torch.stack([entropy_q / (entropy_q + entropy_a)], dim=1)  # Shape: [B, 8]
        return entropy_q

    def normalized_entropy(self, tensor):
        """
        Compute the normalized entropy for a batch of N numbers.

        Args:
            tensor (torch.Tensor): A tensor of shape [B, N, 1] containing non-negative numbers.

        Returns:
            torch.Tensor: A tensor of shape [B] with normalized entropy values in the range [0,1].
        """
        # Remove the last singleton dimension if it exists (from [B, N, 1] to [B, N])
        tensor = tensor.squeeze(-1)

        # Ensure the tensor is a float
        tensor = tensor.float()

        # Compute probability distribution per batch
        prob = tensor / tensor.sum(dim=1, keepdim=True)  # Shape: [B, N]

        # Avoid log(0) issues
        prob = torch.clamp(prob, min=1e-10)

        # Compute entropy: H = -sum(p * log2(p)), summed across N axis
        entropy = -torch.sum(prob * torch.log2(prob), dim=1)  # Shape: [B]

        # Compute max entropy: H_max = log2(N) (same for all batches)
        N = tensor.shape[1]
        max_entropy = torch.log2(torch.tensor(N, dtype=torch.float32, device=tensor.device))

        # Compute normalized entropy
        normalized_H = entropy / max_entropy  # Shape: [B]

        return normalized_H

    def parallel_differentiable_fps_batch_features(self, features, Q, atten, eps=1e-8, M=4):
        """
        Parallel differentiable farthest point sampling.

        Args:
            features: Tensor of shape [B, N, d].
            Q: Quality scores [B, N].
            atten: Attention maps [B, N, N].
            M: Number of parallel samples per iteration.
            self.selection_count: K (total tokens to select).
        """
        scaler = 1e6
        tau = 1e-10 if self.training else 1e-10
        K = self.selection_count
        B, N, d = features.shape

        # ------ Compute attention-based distance ------
        attention_probs = atten / atten.sum(dim=(1, 2), keepdim=True)
        entropy_a = self.normalized_entropy(attention_probs.view(B, -1))
        q_entropy = self.normalized_entropy(Q)
        gamma = self.gamma * (self.gamma1 * q_entropy + (1 - self.gamma1) * entropy_a)

        A = self.cosine_distance_matrix(features, Q.squeeze(), gamma)  # [B, N, N]

        # ------ Initialize selections ------
        Q = Q.squeeze() * scaler
        mask = torch.zeros_like(Q)
        init_logits = Q - mask * 1e6
        first = F.gumbel_softmax(init_logits, tau=tau, hard=True)  # [B, N]

        selections_list = [first]
        mask = mask + first
        selected_set = first.unsqueeze(1)

        # ------ Parallel sampling loop ------
        rounds = (K - 1) // M + 1
        tokens_needed = K - 1  # we already selected 1

        for _ in range(rounds):
            sel_dists = torch.bmm(selected_set, A)  # [B, S, N]
            d = -torch.logsumexp(-sel_dists / tau, dim=1) * tau  # [B, N]
            d = d.masked_fill(mask.bool(), -float('inf'))

            M_now = min(M, tokens_needed)
            g = -torch.empty_like(d).exponential_().log()
            y = (d + g) / tau
            new_ids = y.topk(M_now, dim=1).indices  # [B, M_now]

            for m in range(M_now):
                idx = new_ids[:, m]  # [B]
                one_hot = F.one_hot(idx, N).float()  # [B, N]
                selections_list.append(one_hot)
                selected_set = torch.cat([selected_set, one_hot.unsqueeze(1)], dim=1)
                mask = mask + one_hot

            tokens_needed -= M_now
            if tokens_needed <= 0:
                break

        # ------ Gather selected features ------
        selections = torch.stack(selections_list, dim=1)  # [B, K, N]
        subset_features = torch.bmm(selections, features)  # [B, K, d]
        return subset_features

    import torch

    import torch
    import torch.nn.functional as F

    def jsd_row_distance(self,
            atten: torch.Tensor,  # [B, N, N] row-stochastic (or unnormalized)
            sqrt: bool = True,  # return sqrt(JSD) which is a proper metric
            eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Pairwise Jensen–Shannon divergence between attention rows.

        D[b,i,j] = JSD( P=A[b,i,:], Q=A[b,j,:] )
        where P,Q are row-normalized distributions over N tokens.

        Complexity: O(B * N^3) (pairwise over rows, summing over N columns).
        """
        B, N, _ = atten.shape

        # 1) normalize rows to probabilities
        P = atten / (atten.sum(dim=-1, keepdim=True) + eps)  # [B,N,N]
        P = P.clamp_min(eps)
        P = P / (P.sum(dim=-1, keepdim=True))  # exact renorm

        # 2) per-row entropies H(P)
        HP = -(P * (P + eps).log()).sum(dim=-1)  # [B,N]

        # 3) pairwise mixture M = (P_i + P_j)/2 via broadcasting
        P_i = P.unsqueeze(2)  # [B,N,1,N]
        P_j = P.unsqueeze(1)  # [B,1, N,N]
        M = 0.5 * (P_i + P_j)  # [B,N,N,N]

        # 4) H(M) and JSD = H(M) - 0.5*(H(P_i)+H(P_j))
        HM = -(M * (M + eps).log()).sum(dim=-1)  # [B,N,N]
        JSD = HM - 0.5 * (HP.unsqueeze(2) + HP.unsqueeze(1))  # [B,N,N]
        JSD = torch.clamp(JSD, min=0.0)  # numerical safety

        if sqrt:
            JSD = torch.sqrt(JSD)  # sqrt-JSD is a metric

        # 5) zero self-distance
        eye = torch.eye(N, device=atten.device, dtype=torch.bool).unsqueeze(0)
        JSD = JSD.masked_fill(eye, 0.0)

        return JSD

    def fps_from_inverse_attention(
            self,
            features: torch.Tensor,  # [B, N, D]
            Q: torch.Tensor,  # [B, N] or [B, N, 1]  (used ONLY for seeding)
            atten: torch.Tensor,  # [B, N, N] (head-avg if needed; drop CLS outside)
            tau: float = 0.5,  # anneal during finetune: e.g., 0.7 -> 0.07
            eps: float = 1e-6,
            clamp_max: float = 2000.0,  # cap distances when attn≈0
    ):
        if self.training:
            # tau = 0.5 if (tau is None) else tau
            # scaler = 1.0
            tau = 1e-6
            scaler = 1e6

        else:
            tau = 1e-6
            scaler = 1e6

        """
        Differentiable FPS using ONLY an inverse-attention distance:

          A  = row_norm(atten)                     # [B,N,N]
          A  = 0.5*(A + A^T)                       # symmetrize
          D  = zscore( 1 / (A + eps) ), diag=0     # inverse-attention distance
          seed first token by Q (Gumbel-Top1)
          iterate: soft-min to selected set over D

        Returns:
            subset_features : [B, K, D]
            selections      : [B, K, N]  (ST one-hots)
            aux             : {'D_att': D}
        """

        # ---- helpers to enforce [B,N] logits/one-hots ----
        def _to_BN(x):
            # squeeze a trailing singleton channel dim if present
            return x.squeeze(-1) if (x.dim() == 3 and x.size(-1) == 1) else x

        B, N, D = features.shape
        device = features.device
        K = int(self.selection_count)
        K = min(K, N)

        D_cos = self.jsd_row_distance(atten, sqrt=True)
        if 0:
            if atten is not None:
                # Row L2-normalize attention rows, then cosine sim via row-row dot
                A_row = F.normalize(atten, p=2, dim=-1, eps=eps)  # [B,N,N]
                sim_att = torch.bmm(A_row, A_row.transpose(1, 2)).clamp(-1, 1)  # [B,N,N]
                D_row = 1.0 - sim_att  # [B,N,N]
                # zero diagonal (self-distance)
                eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
                D_row = D_row.masked_fill(eye, 0.0)
                # This is the distance we’ll use
                D_cos = D_row

        # B, N, D = features.shape
        # K = min(int(self.selection_count), N)
        #
        # # ---- 1) Build inverse-attention distance D_att ----
        # X = F.normalize(features, dim=-1, eps=eps)  # [B,N,D]
        # sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)  # [B,N,N]
        # # norm sim to be in [0,1]
        # sim = (sim + 1.0) / 2.0
        # D_cos = 1.0 - sim  # [B,N,N]

        #A = atten / (atten.sum(dim=-1, keepdim=True) + eps)  # row-normalize [B,N,N]
        #A = 0.5 * (A + A.transpose(-1, -2))  # symmetrize
        #D_att = 1.0 / (A + eps)  # inverse-attention
        # zero diagonal
        #eye = torch.eye(N, device=features.device, dtype=torch.bool).unsqueeze(0)
        #D_att = D_att.masked_fill(eye, 0.0)
        # clamp & per-batch standardize for stable logits
        # if clamp_max is not None:
        #     D_att = D_att.clamp_max(clamp_max)
        # mu = D_att.mean(dim=(1, 2), keepdim=True)
        # sd = D_att.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        # D_att = (D_att - mu) / sd  # [B,N,N]
        D_att = D_cos
        # ---- 2) Seed by Q (Gumbel-Top1, straight-through) ----
        Q_ = _to_BN(Q)  # [B,N]
        soft0 = F.gumbel_softmax(Q_ * scaler, tau=tau, hard=False)  # [B,N]
        hard0 = F.gumbel_softmax(Q_ * scaler, tau=tau, hard=True)  # [B,N]
        first = hard0 - soft0.detach() + soft0  # ST one-hot, [B,N]

        selections = [first]  # list of [B,N]
        selected = first.unsqueeze(1)  # [B,1,N] (stack of ST one-hots)
        chosen_mask = first.clone()  # [B,N]

        # ---- 3) Iterative FPS using soft-min over D_att (no argmax, no gather indexing) ----
        for _ in range(1, K):
            # distances from selected set to each candidate:
            #   sel_d[b,s,:] = Σ_i selected[b,s,i] * D_att[b,i,:]
            sel_d = torch.bmm(selected, D_att)  # [B,S,N], fully differentiable

            # soft-min over the selected set
            d_base = -torch.logsumexp(-sel_d / tau, dim=1) * tau  # [B,N]

            # forbid already-selected tokens; use large negative (not -inf) to keep grads finite
            d_masked = d_base.masked_fill(chosen_mask.bool(), -1e9)  # [B,N]

            # next selection (ST)
            soft = F.gumbel_softmax(d_masked * scaler, tau=tau, hard=False)  # [B,N]
            hard = F.gumbel_softmax(d_masked * scaler, tau=tau, hard=True)  # [B,N]
            new = hard - soft.detach() + soft  # [B,N]

            selections.append(new)
            selected = torch.cat([selected, new.unsqueeze(1)], dim=1)  # [B,S+1,N]
            chosen_mask = chosen_mask + new

        # ---- 4) Materialize subset ----
        S = torch.stack(selections, dim=1)  # [B,K,N]
        subset_features = torch.bmm(S, features)  # [B,K,D]

        return subset_features

    import torch
    import torch.nn.functional as F

    import torch
    import torch.nn.functional as F

    import torch
    import torch.nn.functional as F

    def facility_location_select_fixed_balance(
            self,
            features: torch.Tensor,  # [B, N, D]
            Q: torch.Tensor,  # [B, N] or [B, N, 1]  (importance weights)
            atten: torch.Tensor | None = None,  # [B, N, N] if sim_from="attention"
            sim_from: str = "cosine",  # "cosine" | "attention"
            K: int | None = None,
            lambda_qdiv: float = 1.0,  # fixed mix: 1.0=all quality, 0.0=all diversity
            tau_cover: float = 0.7,  # coverage smooth-max temperature
            tau_gumbel: float = 0.7,  # selection temperature
            use_relu_surrogate: bool = True,  # True: hinge/max; False: softplus/LSE
            eps: float = 1e-6,
    ):
        """
        Facility Location with a FIXED balance between quality-weighted coverage and pure diversity:
            gain = lambda_qdiv * gain_qual + (1 - lambda_qdiv) * gain_div

        Coverage surrogate (choose one):
          - hinge : Δ_ij = relu(sim(i,j) - C_j),    coverage C <- max(C, s @ sim)
          - soft  : Δ_ij = tau_cover * softplus((sim(i,j)-C_j)/tau_cover),
                     coverage C <- tau_cover * log( exp(C/tau_cover) + exp((s@sim)/tau_cover) )

        Returns:
            subset_features : [B, K, D]
            selections      : [B, K, N]  (ST one-hots)
            aux             : diagnostics dict
        """

        def _to_bn(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(-1) if (x.dim() == 3 and x.size(-1) == 1) else x

        if self.training:
            # tau = 0.5 if (tau is None) else tau
            # scaler = 1.0

            tau_cover = 0.1
            tau_gumbel = 1e-6
            scaler = 1e6

        else:
            tau_cover = 1e-2
            tau_gumbel = 1e-6
            scaler = 1e6

        B, N, D = features.shape
        if K is None:
            K = int(self.selection_count)
        K = min(K, N)

        # --- similarity sim ∈ [B,N,N] in ~[0,1] ---
        if sim_from == "attention":
            assert atten is not None, "atten must be provided when sim_from='attention'"
            A = atten / (atten.sum(dim=-1, keepdim=True) + eps)
            sim = 0.5 * (A + A.transpose(-1, -2))  # ~[0,1]
        else:
            X = F.normalize(features, dim=-1, eps=eps)
            sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)
            sim = 0.5 * (sim + 1.0)  # map to [0,1]

        # --- quality weights q_j ---
        q = _to_bn(Q).clamp_min(eps)  # [B,N]
        q = q / q.sum(dim=1, keepdim=True)

        # --- coverage state C_j; 0 means "no coverage" with sim∈[0,1] ---
        C = torch.zeros(B, N, device=features.device, dtype=features.dtype)

        selections = []
        chosen_mask = torch.zeros(B, N, device=features.device, dtype=features.dtype)

        # ---- iter 0: seed with quality-weighted coverage Σ_j q_j * sim(i,j) ----
        seed_scores = torch.einsum("bj,bij->bi", q, sim)  # [B,N]
        s = F.gumbel_softmax(seed_scores * scaler, tau=tau_gumbel, hard=True)  # ST one-hot
        selections.append(s)
        chosen_mask = chosen_mask + s

        # coverage update
        sim_new = torch.einsum("bi,bij->bj", s, sim)  # [B,N]
        if use_relu_surrogate:
            C = torch.maximum(C, sim_new)
        else:
            C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # ---- subsequent K-1 selections ----
        for _ in range(1, K):
            # Δ_ij
            if use_relu_surrogate:
                delta = torch.relu(sim - C.unsqueeze(1))  # [B,N,N]
            else:
                L = (sim - C.unsqueeze(1)) / tau_cover
                delta = tau_cover * F.softplus(L)  # [B,N,N]

            #gain = (q.unsqueeze(1) * (tau_cover * F.softplus(L))).sum(dim=-1)  # [B,N]
            # quality-weighted and diversity (unweighted) gains
            gain_qual = (q.unsqueeze(1) * delta).sum(dim=-1)  # [B,N]
            gain_div = delta.mean(dim=-1)  # [B,N]
            gain = lambda_qdiv * gain_qual + (1.0 - lambda_qdiv) * gain_div

            # forbid already-selected tokens
            gain = gain.masked_fill(chosen_mask.bool(), -1e9)

            # stabilize logits a bit
            # gain = gain - gain.mean(dim=1, keepdim=True)
            # gain = gain / (gain.std(dim=1, keepdim=True) + 1e-6)
            # gain = gain.clamp(-10, 10)

            s = F.gumbel_softmax(gain * scaler, tau=tau_gumbel, hard=True)  # ST one-hot
            selections.append(s);
            chosen_mask = chosen_mask + s

            # coverage update
            sim_new = torch.einsum("bi,bij->bj", s, sim)
            if use_relu_surrogate:
                C = torch.maximum(C, sim_new)
            else:
                C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # --- materialize subset ---
        S = torch.stack(selections, dim=1)  # [B,K,N]
        subset_features = torch.bmm(S, features)  # [B,K,D]

        # aux = {
        #     "lambda_qdiv": torch.tensor(lambda_qdiv),
        #     "sim_mean": sim.mean(dim=(1, 2)),
        #     "coverage_mean": C.mean(dim=1),
        #     "surrogate": "relu" if use_relu_surrogate else "softplus",
        #     "tau_cover": torch.tensor(tau_cover),
        #     "tau_gumbel": torch.tensor(tau_gumbel),
        # }
        return subset_features


    def facility_location_select_parallel_hard(
            self,
            features: torch.Tensor,  # [B, N, D]
            Q: torch.Tensor,  # [B, N]
            atten: torch.Tensor | None = None,  # optional [B, N, N]
            sim_from: str = "cosine",
            K: int | None = None,
            lambda_qdiv: float = 1.0,
            tau_cover: float = 0.5,
            tau_select: float = 0.5,
            eps: float = 1e-6,
    ):
        """
        Parallel differentiable Facility Location selector with *hard* (greedy) selection.
        - Builds soft gates in parallel for all K steps.
        - Converts them to distinct one-hot selections (no duplicates).
        """

        def _to_bn(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(-1) if (x.dim() == 3 and x.size(-1) == 1) else x

        B, N, D = features.shape
        if K is None:
            K = int(self.selection_count)
        K = min(K, N)

        # --- similarity matrix ---
        if sim_from == "attention":
            assert atten is not None, "atten must be provided when sim_from='attention'"
            sim = 0.5 * (atten + atten.transpose(-1, -2))
            sim = sim / (sim.sum(dim=-1, keepdim=True) + eps)
        else:
            X = F.normalize(features, dim=-1, eps=eps)
            sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)
            sim = 0.5 * (sim + 1.0)

        # --- normalize quality ---
        q = _to_bn(Q).clamp_min(eps)
        q = q / q.sum(dim=1, keepdim=True)
        sim_q = sim * q.unsqueeze(1)  # [B, N, N]

        # --- parallel soft gates ---
        logits = sim_q.sum(dim=-1)  # [B, N]
        logits = logits.unsqueeze(1).repeat(1, K, 1)  # [B, K, N]
        gates = F.softmax(logits / tau_select, dim=-1)  # [B, K, N]

        # --- soft coverage (for monitoring / training) ---
        sim_sel = torch.einsum("bkn,bnj->bkj", gates, sim)  # [B, K, N]
        C = tau_cover * torch.logsumexp(sim_sel / tau_cover, dim=1)  # [B, N]

        # --- compute total FL gain (optional diagnostic) ---
        gain_qual = (q * C).sum(dim=-1, keepdim=True)
        gain_div = C.mean(dim=-1, keepdim=True)
        total_gain = lambda_qdiv * gain_qual + (1 - lambda_qdiv) * gain_div

        # --- HARDEN selections: greedy suppression across K ---
        logits_base = sim_q.sum(dim=-1)  # [B, N]
        logits_rep = logits_base.unsqueeze(1).repeat(1, K, 1)  # [B, K, N]
        S_hard = []
        chosen_mask = torch.zeros(B, N, device=features.device, dtype=torch.bool)
        for k in range(K):
            step_logits = logits_rep[:, k, :].masked_fill(chosen_mask, -1e9)
            s_k = torch.zeros_like(step_logits)
            top = step_logits.argmax(dim=-1)
            s_k.scatter_(1, top.unsqueeze(-1), 1.0)
            S_hard.append(s_k)
            chosen_mask.scatter_(1, top.unsqueeze(-1), True)

        S_hard = torch.stack(S_hard, dim=1)  # [B, K, N]
        idx = S_hard.argmax(dim=-1)  # [B, K]

        # --- output subset ---
        subset_features = torch.bmm(S_hard, features)  # [B, K, D]

        return subset_features

    def facility_location_select_parallel(
            self,
            features: torch.Tensor,  # [B, N, D]
            Q: torch.Tensor,  # [B, N]
            atten: torch.Tensor | None = None,  # optional [B, N, N]
            sim_from: str = "cosine",
            K: int | None = None,
            lambda_qdiv: float = 1.0,  # quality–diversity trade-off
            tau_cover: float = 0.5,  # smooth coverage temperature
            tau_select: float = 0.5,  # soft selection temperature
            eps: float = 1e-6,
    ):
        """
        Parallel, fully differentiable Facility-Location-style selector.
        Replaces the greedy loop with soft multi-step selection.

        Each of K selection steps is represented as a soft attention distribution.
        The selections are obtained simultaneously through softmax gating.
        """

        def _to_bn(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(-1) if (x.dim() == 3 and x.size(-1) == 1) else x

        B, N, D = features.shape
        if K is None:
            K = int(self.selection_count)
        K = min(K, N)

        # --- compute similarity ---
        if sim_from == "attention":
            assert atten is not None, "atten must be provided when sim_from='attention'"
            sim = 0.5 * (atten + atten.transpose(-1, -2))
            sim = sim / (sim.sum(dim=-1, keepdim=True) + eps)
        else:
            X = F.normalize(features, dim=-1, eps=eps)
            sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)
            sim = 0.5 * (sim + 1.0)

        # --- normalize quality ---
        q = _to_bn(Q).clamp_min(eps)
        q = q / q.sum(dim=1, keepdim=True)

        # --- precompute q-weighted sim ---
        sim_q = sim * q.unsqueeze(1)  # [B, N, N]

        # --- learnable "step gates" for parallel selection ---
        #   Each gate produces a soft distribution over tokens (K×N)
        logits = sim_q.sum(dim=-1)  # [B, N] initial scores
        logits = logits.unsqueeze(1).repeat(1, K, 1)  # [B, K, N]
        gates = F.softmax(logits / tau_select, dim=-1)  # [B, K, N], soft selections

        # --- aggregate coverage in parallel ---
        # For each token j, compute coverage from all gates
        # C_j = softmax-approx of max_i sim(i,j) over all gates
        sim_sel = torch.einsum("bkn,bnij->bkj", gates, sim)  # [B, K, N]
        C = tau_cover * torch.logsumexp(sim_sel / tau_cover, dim=1)  # [B, N]

        # --- compute gains (quality + diversity) ---
        gain_qual = (q * C).sum(dim=-1, keepdim=True)  # [B,1]
        gain_div = C.mean(dim=-1, keepdim=True)  # [B,1]
        total_gain = lambda_qdiv * gain_qual + (1 - lambda_qdiv) * gain_div

        # --- output subset features ---
        subset_features = torch.bmm(gates, features)  # [B,K,D]

        # aux = {
        #     "gates": gates,  # soft selections
        #     "coverage": C,  # soft coverage per token
        #     "gain_total": total_gain,  # FL objective value
        # }
        return subset_features

    import torch
    import torch.nn.functional as F
    import math

    def facility_location_select_fixed_balance_linear(
            self,
            features: torch.Tensor,  # [B, N, D]
            Q: torch.Tensor,  # [B, N] or [B, N, 1]
            atten: torch.Tensor | None = None,  # [B, N, N] if sim_from="attention"
            sim_from: str = "cosine",  # "cosine" | "attention"
            K: int | None = None,
            lambda_qdiv: float = 0.5,  # balance between quality & diversity in gain
            tau_cover: float = 0.7,  # smooth-max temperature
            tau_gumbel: float = 0.7,  # Gumbel selection temperature
            use_relu_surrogate: bool = True,  # hinge vs. softplus/LSE
            eps: float = 1e-6,
    ):
        """
        Facility-Location token selection with *linear* quality–diversity blending:

            w_ij = (1 − γ) * sim_ij + γ * q_j

        where γ = q_scale is a learnable scalar in [0,1].
        """

        def _to_bn(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(-1) if (x.dim() == 3 and x.size(-1) == 1) else x

        # --- temperatures and scaling ---
        if self.training:
            tau_cover = 0.1
            tau_gumbel = 1e-6
            scaler = 1e6
        else:
            tau_cover = 1e-2
            tau_gumbel = 1e-6
            scaler = 1e6

        B, N, D = features.shape
        if K is None:
            K = int(self.selection_count)
        K = min(K, N)

        # --- similarity matrix sim ∈ [B,N,N] ---
        if sim_from == "attention":
            assert atten is not None, "atten must be provided when sim_from='attention'"
            A = atten / (atten.sum(dim=-1, keepdim=True) + eps)
            sim = 0.5 * (A + A.transpose(-1, -2))
        else:
            X = F.normalize(features, dim=-1, eps=eps)
            sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)
            sim = 0.5 * (sim + 1.0)  # map cosine [-1,1] → [0,1]

        # --- quality weights q ---
        q = _to_bn(Q).clamp_min(eps)
        q = q / q.sum(dim=1, keepdim=True)  # normalize to [B,N]

        # --- linear quality–diversity weighting ---
        # (define once in __init__: self.q_scale_raw = nn.Parameter(torch.tensor(0.0)))
        gamma = torch.sigmoid(self.gamma)  # [0,1]
        w = (1.0 - gamma) * sim + gamma * q.unsqueeze(1)  # [B,N,N]

        # --- precompute caches ---
        sim_mean = sim.mean(dim=-1)  # [B, N]
        C = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        chosen_mask = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        selections = []

        # ---- Step 1: seeding with weighted similarity ----
        seed_scores = w.sum(dim=-1)  # Σ_j w_ij
        s = F.gumbel_softmax(seed_scores * scaler, tau=tau_gumbel, hard=True)
        selections.append(s)
        chosen_mask = chosen_mask + s

        # ---- Coverage update ----
        sim_new = torch.einsum("bi,bij->bj", s, sim)
        if use_relu_surrogate:
            C = torch.maximum(C, sim_new)
        else:
            C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # ---- Step 2..K: iterative selection ----
        for _ in range(1, K):
            if use_relu_surrogate:
                delta = torch.relu(sim - C.unsqueeze(1))
            else:
                L = (sim - C.unsqueeze(1)) / tau_cover
                delta = tau_cover * F.softplus(L)

            # --- gains using the same linear mix ---
            gain_qual = (delta * (gamma * q.unsqueeze(1))).sum(dim=-1)
            gain_div = (delta * ((1.0 - gamma) * torch.ones_like(sim))).mean(dim=-1)
            gain = lambda_qdiv * gain_qual + (1.0 - lambda_qdiv) * gain_div

            gain = gain.masked_fill(chosen_mask.bool(), -1e9)

            s = F.gumbel_softmax(gain * scaler, tau=tau_gumbel, hard=True)
            selections.append(s)
            chosen_mask = chosen_mask + s

            sim_new = torch.einsum("bi,bij->bj", s, sim)
            if use_relu_surrogate:
                C = torch.maximum(C, sim_new)
            else:
                C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # --- Output subset ---
        S = torch.stack(selections, dim=1)  # [B,K,N]
        subset_features = torch.bmm(S, features)  # [B,K,D]

        # --- Diagnostics ---
        # aux = {
        #     "gamma": gamma.detach(),
        #     "coverage_mean": C.mean(),
        #     "sim_mean": sim.mean(),
        # }

        return subset_features

    def facility_location_select_fixed_balance_exp_mix(
            self,
            features: torch.Tensor,  # [B, N, D]
            Q: torch.Tensor,  # [B, N] or [B, N, 1]
            atten: torch.Tensor | None = None,  # [B, N, N] if sim_from="attention"
            sim_from: str = "cosine",  # "cosine" | "attention"
            K: int | None = None,
            lambda_qdiv: float = 1.0,  # balance between quality/diversity in gain
            tau_cover: float = 0.7,  # smooth-max temperature
            tau_gumbel: float = 0.7,  # Gumbel selection temperature
            use_relu_surrogate: bool = True,  # hinge vs soft surrogate
            eps: float = 1e-6,
    ):
        """
        Facility-Location token selection with geometric mean weighting:
            sim_q = sim^(1 - q_scale) * q^(q_scale)

        This yields a smooth multiplicative interpolation between
        geometry-based coverage and importance-weighted coverage.
        """

        def _to_bn(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(-1) if (x.dim() == 3 and x.size(-1) == 1) else x

        # --- temperature setup ---
        if self.training:
            tau_cover = 0.1
            tau_gumbel = 1e-6
            scaler = 1e6
        else:
            tau_cover = 1e-2
            tau_gumbel = 1e-6
            scaler = 1e6

        B, N, D = features.shape
        if K is None:
            K = int(self.selection_count)
        K = min(K, N)

        # --- similarity matrix sim ∈ [B,N,N] ---
        if sim_from == "attention":
            assert atten is not None, "atten must be provided when sim_from='attention'"
            A = atten / (atten.sum(dim=-1, keepdim=True) + eps)
            sim = 0.5 * (A + A.transpose(-1, -2))
        else:
            X = F.normalize(features, dim=-1, eps=eps)
            sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)
            sim = 0.5 * (sim + 1.0)

        # --- quality weights ---
        q = _to_bn(Q).clamp_min(eps)
        q = q / q.sum(dim=1, keepdim=True)  # normalize to [B,N]

        # --- exponent-based weighting ---
        # define once in __init__: self.q_scale = torch.nn.Parameter(torch.tensor(0.5))
        gamma = self.gamma.sigmoid()
        #gamma = torch.clamp(self.q_scale, 0.0, 1.0)
        sim_q = (sim.clamp_min(eps) ** (1 - gamma)) * (q.unsqueeze(1).clamp_min(eps) ** gamma)

        # --- precompute caches ---
        sim_mean = sim.mean(dim=-1)  # [B, N]
        C = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        chosen_mask = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        selections = []

        # ---- Step 1: seeding with quality coverage ----
        seed_scores = sim_q.sum(dim=-1)  # Σ_j w_ij
        s = F.gumbel_softmax(seed_scores * scaler, tau=tau_gumbel, hard=True)
        selections.append(s)
        chosen_mask = chosen_mask + s

        # ---- Coverage update ----
        sim_new = torch.einsum("bi,bij->bj", s, sim)
        if use_relu_surrogate:
            C = torch.maximum(C, sim_new)
        else:
            C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # ---- Step 2..K: iterative selection ----
        for _ in range(1, K):
            if use_relu_surrogate:
                delta = torch.relu(sim - C.unsqueeze(1))
            else:
                L = (sim - C.unsqueeze(1)) / tau_cover
                delta = tau_cover * F.softplus(L)

            # --- gain computation with geometric weighting ---
            gain_qual = (delta * (q.unsqueeze(1) ** gamma)).sum(dim=-1)
            gain_div = (delta * (sim.clamp_min(eps) ** (1 - gamma))).mean(dim=-1)
            gain = lambda_qdiv * gain_qual + (1 - lambda_qdiv) * gain_div

            gain = gain.masked_fill(chosen_mask.bool(), -1e9)

            s = F.gumbel_softmax(gain * scaler, tau=tau_gumbel, hard=True)
            selections.append(s)
            chosen_mask = chosen_mask + s

            sim_new = torch.einsum("bi,bij->bj", s, sim)
            if use_relu_surrogate:
                C = torch.maximum(C, sim_new)
            else:
                C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # --- Output subset ---
        S = torch.stack(selections, dim=1)  # [B,K,N]
        subset_features = torch.bmm(S, features)  # [B,K,D]

        # --- Diagnostics ---
        # aux = {
        #     "q_scale": gamma.detach(),
        #     "coverage_mean": C.mean(),
        #     "sim_mean": sim.mean(),
        # }

        return subset_features

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    def facility_location_select_fixed_balance_adaptive_gamma(
            self,
            features: torch.Tensor,  # [B, N, D]
            Q: torch.Tensor,  # [B, N] or [B, N, 1]
            atten: torch.Tensor | None = None,  # [B, N, N] if sim_from="attention"
            sim_from: str = "cosine",  # "cosine" | "attention"
            K: int | None = None,
            lambda_qdiv: float = 0.5,  # balance between quality & diversity
            tau_cover: float = 0.7,  # smooth-max temperature
            tau_gumbel: float = 0.7,  # Gumbel selection temperature
            use_relu_surrogate: bool = True,  # hinge vs. softplus/LSE
            eps: float = 1e-6,
    ):
        """
        Facility-Location token selection with an *adaptive per-image gamma*
        predicted from attention and token statistics.

            γ_b = f_MLP([entropy(Q_b), mean(sim_b), var(sim_b)])

        Then the blended weight is:
            w_ij = (1 - γ_b) * sim_ij + γ_b * q_j
        """

        def _to_bn(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(-1) if (x.dim() == 3 and x.size(-1) == 1) else x

        # --- temperatures and scaling ---
        if self.training:
            tau_cover = 0.1
            tau_gumbel = 1e-6
            scaler = 1e6
        else:
            tau_cover = 1e-2
            tau_gumbel = 1e-6
            scaler = 1e6

        B, N, D = features.shape
        if K is None:
            K = int(self.selection_count)
        K = min(K, N)

        # --- similarity matrix sim ∈ [B,N,N] ---
        if sim_from == "attention":
            assert atten is not None, "atten must be provided when sim_from='attention'"
            A = atten / (atten.sum(dim=-1, keepdim=True) + eps)
            sim = 0.5 * (A + A.transpose(-1, -2))
        else:
            X = F.normalize(features, dim=-1, eps=eps)
            sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)
            sim = 0.5 * (sim + 1.0)  # map cosine [-1,1] → [0,1]

        # --- quality weights q ---
        q = _to_bn(Q).clamp_min(eps)
        q = q / q.sum(dim=1, keepdim=True)  # [B,N]

        # --- compute per-image statistics for adaptive gamma ---
        # 1. entropy of attention/quality
        H_q = -(q * (q.clamp_min(1e-8)).log()).sum(dim=1, keepdim=True)
        H_q = H_q / math.log(q.size(1) + 1e-8)  # normalized [0,1]
        # 2. mean & variance of similarity
        sim_mean = sim.mean(dim=(1, 2), keepdim=True)  # [B,1,1]
        sim_var = sim.var(dim=(1, 2), keepdim=True)  # [B,1,1]

        # Combine into feature vector [B,3]
        stats = torch.cat([H_q, sim_mean.squeeze(-1), sim_var.squeeze(-1)], dim=1)  # [B,3]

        # --- Predict γ per image ---
        # define once in __init__:
        # self.gamma_head = nn.Sequential(
        #     nn.Linear(3, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 1),
        #     nn.Sigmoid()
        # )
        gamma = self.gamma_head(stats).view(B, 1, 1)  # [B,1,1] ∈ (0,1)

        # --- linear blend for sim/q per image ---
        w = (1.0 - gamma) * sim + gamma * q.unsqueeze(1)  # [B,N,N]

        # --- precompute caches ---
        sim_mean = sim.mean(dim=-1)  # [B, N]
        C = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        chosen_mask = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        selections = []

        # ---- Step 1: seeding ----
        seed_scores = w.sum(dim=-1)  # Σ_j w_ij
        s = F.gumbel_softmax(seed_scores * scaler, tau=tau_gumbel, hard=True)
        selections.append(s)
        chosen_mask += s

        # ---- Coverage update ----
        sim_new = torch.einsum("bi,bij->bj", s, sim)
        if use_relu_surrogate:
            C = torch.maximum(C, sim_new)
        else:
            C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # ---- Step 2..K: iterative selection ----
        for _ in range(1, K):
            if use_relu_surrogate:
                delta = torch.relu(sim - C.unsqueeze(1))
            else:
                L = (sim - C.unsqueeze(1)) / tau_cover
                delta = tau_cover * F.softplus(L)

            # --- gain computation ---
            gain_qual = (delta * (gamma * q.unsqueeze(1))).sum(dim=-1)
            gain_div = (delta * ((1.0 - gamma) * torch.ones_like(sim))).mean(dim=-1)
            gain = lambda_qdiv * gain_qual + (1.0 - lambda_qdiv) * gain_div

            gain = gain.masked_fill(chosen_mask.bool(), -1e9)

            s = F.gumbel_softmax(gain * scaler, tau=tau_gumbel, hard=True)
            selections.append(s)
            chosen_mask += s

            sim_new = torch.einsum("bi,bij->bj", s, sim)
            if use_relu_surrogate:
                C = torch.maximum(C, sim_new)
            else:
                C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # --- Output subset ---
        S = torch.stack(selections, dim=1)  # [B,K,N]
        subset_features = torch.bmm(S, features)  # [B,K,D]

        # # --- Diagnostics ---
        # aux = {
        #     "gamma_mean": gamma.mean().item(),
        #     "gamma_per_image": gamma.detach().squeeze(-1).cpu(),
        #     "entropy_mean": H_q.mean().item(),
        #     "sim_mean": sim_mean.mean().item(),
        # }

        return subset_features

    def select_tokens_multiplicative(
            self,
            features: torch.Tensor,  # [B, N, D]
            Q: torch.Tensor,  # [B, N]
            sim: torch.Tensor,  # [B, N, N] similarity matrix in [0,1]
            a: float = 1.0,  # exponent for quality
            b: float = 1.0,  # exponent for coverage/novelty
            tau_g: float = 1e-6,  # Gumbel temperature
            eps: float = 1e-6,
    ):
        """
        Multiplicative Quality–Coverage Selector:
            gain(i) = (Q_i_norm ** a) * (Delta_i_norm ** b)

        Where:
          Q_i_norm     = unary importance of token i
          Delta_i_norm = total novel coverage if selecting token i
        """
        # ------------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------------
        B, N, D = features.shape
        K = int(self.selection_count)
        K = min(K, N)

        device = features.device
        dtype = features.dtype

        # Normalize Q to [0,1]
        Q = Q.squeeze()
        Qn = Q / (Q.max(dim=1, keepdim=True).values + eps)  # [B,N]

        # Coverage vector C[j] = how well token j is already covered
        C = torch.zeros(B, N, device=device, dtype=dtype)  # [B,N]
        chosen_mask = torch.zeros(B, N, device=device, dtype=dtype)

        selections = []

        # ------------------------------------------------------------------
        # 1) First token = highest quality (via gumbel)
        # ------------------------------------------------------------------
        s = F.gumbel_softmax(Qn * 1e6, tau=tau_g, hard=True)  # [B,N]
        selections.append(s)
        chosen_mask += s

        # Update coverage
        sim_new = torch.einsum("bi,bij->bj", s, sim)  # [B,N]
        C = torch.maximum(C, sim_new)

        # ------------------------------------------------------------------
        # 2) Next K-1 picks using multiplicative gain
        # ------------------------------------------------------------------
        for _ in range(1, K):
            # Coverage gain delta[i,j] = max(0, sim(i,j) - C[j])
            delta = torch.relu(sim - C.unsqueeze(1))  # [B,N,N]


            # Gain computations (using cached sim_q and sim_mean)
            gain = (( delta ** a ) * Qn.unsqueeze(1) ** b ).sum(dim=-1)  # [B,N]
            # gain_div = (delta.mean(dim=-1) + sim_mean) / 2.0  # optional regularization
            # gain = lambda_qdiv * gain_qual + (1 - lambda_qdiv) * gain_div
            #
            # gain = gain.masked_fill(chosen_mask.bool(), -1e9)
            #
            # s = F.gumbel_softmax(gain * scaler, tau=tau_gumbel, hard=True)


            #####




            # Token novelty score Δ_i = sum_j delta(i,j)
            # Delta = delta.sum(dim=-1)  # [B,N]
            # Delta = Delta / (Delta.max(dim=1, keepdim=True).values + eps)

            # --------------------------------------------------------------
            # Multiplicative gain:  gain_i = (Q_i^a) * (Δ_i^b)
            # --------------------------------------------------------------
            #gain = (Qn.clamp_min(eps) ** a) * (Delta.clamp_min(eps) ** b)  # [B,N]

            # Prevent reselection
            gain = gain.masked_fill(chosen_mask.bool(), -1e9)

            # Select next token
            s = F.gumbel_softmax(gain * 1e6, tau=tau_g, hard=True)
            selections.append(s)
            chosen_mask += s

            # Update coverage
            sim_new = torch.einsum("bi,bij->bj", s, sim)
            C = torch.maximum(C, sim_new)

        # ------------------------------------------------------------------
        # 3) Produce output subset
        # ------------------------------------------------------------------
        S = torch.stack(selections, dim=1)  # [B,K,N]
        subset = torch.bmm(S, features)  # [B,K,D]

        return subset
    def select_tokens_weighted_gain(
            self,
            features: torch.Tensor,  # [B, N, D]
            Q: torch.Tensor,  # [B, N]
            sim: torch.Tensor,  # [B, N, N] similarity in [0,1]
            alpha: float = 0.45,  # global weighting; if None → use balancer
            tau_g: float = 1e-6,  # Gumbel temperature
            tau_cover: float = 0.1,  # coverage softmax temperature
            eps: float = 1e-6,
    ):
        """
        Balanced Quality + Coverage selection:
            gain(i) = α * Q_i + (1 - α) * Δ_i

        where Δ_i = sum_j max(0, sim(i,j) - C_j)
        and C_j tracks coverage of each token j by selected set S.
        """
        B, N, D = features.shape
        K = int(self.selection_count)
        K = min(K, N)

        device = features.device
        dtype = features.dtype

        # -----------------------------------------------------------
        # 0. Normalize Q to [0,1] per image for stability
        # -----------------------------------------------------------
        Q = Q.squeeze()
        Qn = Q / (Q.max(dim=1, keepdim=True).values + eps)  # [B, N]

        # -----------------------------------------------------------
        # 1. Determine α (quality weight)
        # -----------------------------------------------------------
        if alpha is None:
            # Use balancer to predict α per image
            assert hasattr(self, "balancer"), "No balancer and alpha=None"
            stats = torch.stack([
                Qn.mean(dim=1),
                Qn.std(dim=1),
                Qn.max(dim=1).values,
            ], dim=1)  # [B,3]

            alpha_b = torch.sigmoid(self.balancer(stats)).squeeze(1)  # [B]
        else:
            alpha_b = torch.full((B,), float(alpha), device=device)  # [B]

        alpha_b = alpha_b.view(B, 1)  # [B,1]

        # -----------------------------------------------------------
        # 2. Storage: coverage C_j and chosen mask
        # -----------------------------------------------------------
        C = torch.zeros(B, N, device=device, dtype=dtype)  # [B,N]
        chosen_mask = torch.zeros(B, N, device=device, dtype=dtype)

        selections = []

        # -----------------------------------------------------------
        # 3. First selection: purely highest-Q (via Gumbel)
        # -----------------------------------------------------------
        first = F.gumbel_softmax(Qn * 1e6, tau=tau_g, hard=True)  # [B,N]
        selections.append(first)
        chosen_mask += first

        # Update coverage:
        sim_new = torch.einsum("bi,bij->bj", first, sim)  # [B,N]
        C = torch.maximum(C, sim_new)

        selected = first.unsqueeze(1)  # [B,1,N]

        # -----------------------------------------------------------
        # 4. Next K-1 selections
        # -----------------------------------------------------------
        for _ in range(1, K):
            # Δ_ij = max(0, sim(i,j) - C[j])
            delta = torch.relu(sim - C.unsqueeze(1))  # [B,N,N]

            # Δ_i = sum_j Δ_ij
            Delta = delta.sum(dim=-1)  # [B,N]
            Delta = Delta / (Delta.max(dim=1, keepdim=True).values + eps)

            # -------------------------------------------------------
            # Gain = α * Q + (1 - α) * Δ
            # -------------------------------------------------------
            gain = alpha_b * Qn + (1 - alpha_b) * Delta  # [B,N]

            # Prevent reselection
            gain = gain.masked_fill(chosen_mask.bool(), -1e9)

            # Select next token
            s = F.gumbel_softmax(gain * 1e6, tau=tau_g, hard=True)  # [B,N]
            selections.append(s)
            chosen_mask += s

            # Update coverage
            sim_new = torch.einsum("bi,bij->bj", s, sim)  # [B,N]
            C = torch.maximum(C, sim_new)

            selected = torch.cat([selected, s.unsqueeze(1)], dim=1)

        # -----------------------------------------------------------
        # 5. Convert to actual feature subset
        # -----------------------------------------------------------
        S = torch.stack(selections, dim=1)  # [B,K,N]
        subset = torch.bmm(S, features)  # [B,K,D]

        return subset

    def facility_location_select_solution3_centrality_balance_v2(
            self,
            features: torch.Tensor,  # [B, N, D]
            att_cls: torch.Tensor,  # [B, N]
            atten: torch.Tensor | None,  # [B, N, N]
            sim_from: str = "cosine",
            K: int | None = None,
            tau_cover: float = 0.1,
            tau_gumbel: float = 0.5,  # will be overridden by schedule below
            eps: float = 1e-6,
            q_floor: float = 0.05,  # small floor for stability (tune 0.02-0.1)
            use_q_weighted_gain: bool = True,
    ):
        B, N, D = features.shape

        # -----------------------------
        # 0) temps / scaler
        # -----------------------------
        # If you truly want hard greedy, keep tau_gumbel tiny.
        # If you want lambda/alpha to learn, anneal tau_gumbel.
        if self.training:
            tau_cover = 0.1
            # Suggest: start ~0.5 and anneal in training loop; here just a default.
            tau_gumbel = 1e-6
            scaler = 1e6
        else:
            tau_cover = 1e-2
            tau_gumbel = 1e-6
            scaler = 1e6

        # -----------------------------
        # 1) K
        # -----------------------------
        if K is None:
            K = int(self.selection_count)
        K = min(K, N)

        # -----------------------------
        # 2) similarity sim in [0,1]
        # -----------------------------
        X = F.normalize(features, dim=-1, eps=eps)
        sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)
        sim = 0.5 * (sim + 1.0)

        # -----------------------------
        # 3) importance Q: mix attention + centrality
        # -----------------------------
        att_cls = att_cls.view(B, N)

        if atten is None:
            in_central = sim.mean(dim=1)  # [B,N]
        else:
            in_central = atten.mean(dim=1)  # [B,N]

        a_tilde = att_cls / (att_cls.sum(dim=-1, keepdim=True) + eps)
        z_tilde = in_central / (in_central.sum(dim=-1, keepdim=True) + eps)

        lam = torch.sigmoid(self.gamma)  # scalar or per-layer param
        Q = lam * a_tilde + (1.0 - lam) * z_tilde  # [B,N], sums to 1

        # normalized Q used as weights in quality-weighted coverage
        q = Q.clamp_min(eps)
        q = q / q.sum(dim=1, keepdim=True)  # [B,N], sums to 1

        # scale Q to have mean ~1 (stabilizes multiplicative gate)
        Qn = Q * float(N)  # [B,N], mean ~1

        alpha = F.softplus(self.beta)  # >=0
        gate = q_floor + alpha * Qn  # [B,N], stable positive

        # -----------------------------
        # 4) coverage state + masks
        # -----------------------------
        C = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        chosen_mask = torch.zeros_like(C)
        selections = []

        # -----------------------------
        # 5) seed: quality-weighted coverage (important)
        # -----------------------------
        seed_scores = torch.einsum("bj,bij->bi", q, sim)  # [B,N]
        s = F.gumbel_softmax(seed_scores * scaler, tau=tau_gumbel, hard=True)
        selections.append(s)
        chosen_mask += s

        sim_new = torch.einsum("bi,bij->bj", s, sim)
        C = torch.maximum(C, sim_new)

        # -----------------------------
        # 6) greedy steps
        # -----------------------------
        for _ in range(1, K):
            delta = torch.relu(sim - C.unsqueeze(1))  # [B,N,N]

            if use_q_weighted_gain:
                # quality-weighted marginal gain (more stable, matches your best variant)
                gain_base = (q.unsqueeze(1) * delta).sum(dim=-1)  # [B,N]
            else:
                # classic FL marginal gain
                gain_base = delta.sum(dim=-1)  # [B,N]

            # gate biases but does not replace coverage
            gain = gain_base * gate  # [B,N]

            gain = gain.masked_fill(chosen_mask.bool(), -1e9)

            s = F.gumbel_softmax(gain * scaler, tau=tau_gumbel, hard=True)
            selections.append(s)
            chosen_mask += s

            sim_new = torch.einsum("bi,bij->bj", s, sim)
            C = torch.maximum(C, sim_new)

        S = torch.stack(selections, dim=1)  # [B,K,N]
        subset_features = torch.bmm(S, features)  # [B,K,D]
        return subset_features

    def facility_location_select_solution3_centrality_balance(
            self,
            features: torch.Tensor,  # [B, N, D]
            att_cls: torch.Tensor,  # [B, N]
            atten: torch.Tensor | None,  # [B, N, N], attention matrix (avg over heads)
            sim_from: str = "cosine",
            K: int | None = None,
            tau_cover: float = 0.3,
            tau_gumbel: float = 0.3,
            eps: float = 1e-6,
    ):
        """
        FACILITY LOCATION (Solution 3, with learned balances)
        -----------------------------------------------------
        - Token importance Q_i combines CLS relevance and attention-based centrality:
              Q_i = λ * ã_i + (1 - λ) * z̃_i,
            where λ = sigmoid(gamma) is learned and both components are normalized.

        - Facility Location gain combines coverage and importance:
              gain(i) = Δ(i) * (1 + α * Q_i),
            where α = softplus(beta) is learned and Q_i is set-independent,
            so FL remains monotone submodular.

        - Greedy FL selection is implemented with a straight-through Gumbel-softmax.
        """

        B, N, D = features.shape

        # ------------------------------------------------------------------
        # 0) Training / inference temperatures & scaler
        # ------------------------------------------------------------------
        if self.training:
            tau_cover = 0.1
            tau_gumbel = 1e-6
            scaler = 1e6
        else:
            tau_cover = 1e-2
            tau_gumbel = 1e-6
            scaler = 1e6

        # ------------------------------------------------------------------
        # 1) Number of tokens to select
        # ------------------------------------------------------------------
        if K is None:
            K = int(self.selection_count)
        K = min(K, N)

        # ------------------------------------------------------------------
        # 2) Similarity matrix (cosine in [0, 1])
        # ------------------------------------------------------------------
        X = F.normalize(features, dim=-1, eps=eps)  # [B, N, D]
        sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)  # [B, N, N]
        sim = 0.5 * (sim + 1.0)  # [-1,1] -> [0,1]
        sim = sim.clamp(0.0, 1.0)

        # ------------------------------------------------------------------
        # 3) Token importance Q_i: learned balance of CLS relevance and centrality
        # ------------------------------------------------------------------
        # att_cls: [B, N]
        att_cls = att_cls.view(B, N)

        if atten is None:
            # Fallback: use similarity-based centrality if attention is not provided
            # (you can remove this branch if atten is always given)
            in_central = sim.mean(dim=1)  # [B, N], avg over "j": proxy centrality
        else:
            # atten: [B, N, N], average incoming attention mass per token
            # Assume atten[b, j, i] ~ attention from token j to token i
            in_central = atten.mean(dim=1)  # [B, N], avg over all senders j

        # Normalize both components across tokens (per batch) for stable mixing
        a_tilde = att_cls / (att_cls.sum(dim=-1, keepdim=True) + eps)  # [B, N]
        z_tilde = in_central / (in_central.sum(dim=-1, keepdim=True) + eps)  # [B, N]

        # Learned mixture weight λ = sigmoid(gamma) \in (0,1)
        lam = torch.sigmoid(self.gamma)

        # Token importance Q_i
        Q = lam * a_tilde + (1.0 - lam) * z_tilde  # [B, N]
        # normalized Q used as weights in quality-weighted coverage
        q = Q.clamp_min(eps)
        q = q / q.sum(dim=1, keepdim=True)  # [B,N], sums to 1

        # scale Q to have mean ~1 (stabilizes multiplicative gate)
        Qn = Q * float(N)
        # Learned strength α = softplus(beta) ≥ 0
        alpha = F.softplus(self.beta)

        # Importance modulation factor: (1 + α Q_i)
        # Shape: [B, N], will broadcast over Δ(i)
        #quality_factor = 1.0 + alpha * Q

        q_floor = 0.05
        Qn = Q * N
        quality_factor = q_floor + alpha * Qn

        #quality_factor = alpha * Q

        # ------------------------------------------------------------------
        # 4) Coverage accumulator C (initially empty) and chosen mask
        # ------------------------------------------------------------------
        C = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        chosen_mask = torch.zeros_like(C)  # [B, N]
        selections = []

        # ------------------------------------------------------------------
        # 5) Seed selection: pure coverage (no Q influence)
        # ------------------------------------------------------------------
        seed_scores = sim.sum(dim=-1)  # [B, N], Δ-like initial score
        s = F.gumbel_softmax(seed_scores * scaler, tau=tau_gumbel, hard=True)
        selections.append(s)
        chosen_mask += s

        # Update coverage C using selected seed
        sim_new = torch.einsum("bi,bij->bj", s, sim)  # [B, N]
        C = torch.maximum(C, sim_new)

        # ------------------------------------------------------------------
        # 6) Greedy Facility Location iterations with importance modulation
        # ------------------------------------------------------------------
        for _ in range(1, K):
            # Δ_ij = relu(sim(i,j) - C_j)
            delta = torch.relu(sim - C.unsqueeze(1))  # [B, N, N]

            # Δ(i) = Σ_j Δ_ij
            delta_sum = delta.sum(dim=-1)  # [B, N]

            # gain(i) = Δ(i) * (1 + α * Q_i)
            gain = delta_sum * quality_factor  # [B, N]

            # mask already chosen tokens
            gain = gain.masked_fill(chosen_mask.bool(), -1e9)

            # soft argmax via straight-through Gumbel-softmax
            s = F.gumbel_softmax(gain * scaler, tau=tau_gumbel, hard=True)
            selections.append(s)
            chosen_mask += s

            # update coverage after selecting i*
            sim_new = torch.einsum("bi,bij->bj", s, sim)  # [B, N]
            C = torch.maximum(C, sim_new)

        # ------------------------------------------------------------------
        # 7) Output reduced feature set
        # ------------------------------------------------------------------
        S = torch.stack(selections, dim=1)  # [B, K, N]
        subset_features = torch.bmm(S, features)  # [B, K, D]

        return subset_features

    def facility_location_select_solution3_centrality(
            self,
            features: torch.Tensor,  # [B, N, D]
            att_cls: torch.Tensor,  # [B, N]
            atten: torch.Tensor | None = None,  # [B, N, N] if sim_from="attention"
            sim_from: str = "cosine",
            K: int | None = None,
            tau_cover: float = 0.3,
            tau_gumbel: float = 0.3,
            eps: float = 1e-6,
    ):
        """
        FACILITY LOCATION (Solution 3)
        --------------------------------------
        Pure Δ-based diversity + soft quality modulation:
            gain(i) = Δ(i) * (1 + η * Q_norm(i))

        - No Top-K
        - No exponent weighting
        - Only ONE selection mechanism (FL)
        - Smooth, stable, numerically safe
        - sim is computed internally as in your original implementation
        """

        B, N, D = features.shape

        if self.training:
            tau_cover = 0.1
            tau_gumbel = 1e-6
            scaler = 1e6
        else:
            tau_cover = 1e-2
            tau_gumbel = 1e-6
            scaler = 1e6


        if K is None:
            K = int(self.selection_count)
        K = min(K, N)


        X = F.normalize(features, dim=-1, eps=eps)
        sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)
        sim = 0.5 * (sim + 1.0)  # map [-1,1] -> [0,1]
        sim = sim.clamp(0.0, 1.0)


        att_cls = att_cls.squeeze()
        in_central = atten.mean(dim=1)
        beta = 0.5
        Q =  beta * att_cls +  in_central

        quality_factor = self.q_exp * Q  # [B, N]

        # ----------------------------------------------------------
        # 4) Coverage accumulator C (initially empty)
        # ----------------------------------------------------------
        C = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        chosen_mask = torch.zeros_like(C)
        selections = []

        # ----------------------------------------------------------
        # 5) Seed selection (Δ-like, no Q influence yet)
        # ----------------------------------------------------------
        seed_scores = sim.sum(dim=-1)  # [B, N]
        s = F.gumbel_softmax(seed_scores * scaler, tau=tau_gumbel, hard=True)

        selections.append(s)
        chosen_mask += s

        # update coverage C
        sim_new = torch.einsum("bi,bij->bj", s, sim)
        C = torch.maximum(C, sim_new)

        # ----------------------------------------------------------
        # 6) Greedy Facility Location iterations
        # ----------------------------------------------------------
        for _ in range(1, K):
            # Δ_ij = relu(sim(i,j) - C_j)
            delta = torch.relu(sim - C.unsqueeze(1))  # [B, N, N]

            # Δ(i) = Σ_j Δ_ij
            delta_sum = delta.sum(dim=-1)  # [B, N]

            # Unified single-mechanism gain:
            #     gain(i) = Δ(i) * (1 + η * Q_norm(i))
            gain = delta_sum * quality_factor  # [B, N]

            # mask already chosen tokens
            gain = gain.masked_fill(chosen_mask.bool(), -1e9)

            # soft argmax
            s = F.gumbel_softmax(gain * scaler, tau=tau_gumbel, hard=True)
            selections.append(s)
            chosen_mask += s

            # update coverage after selecting i*
            sim_new = torch.einsum("bi,bij->bj", s, sim)
            C = torch.maximum(C, sim_new)

        # ----------------------------------------------------------
        # 7) Output reduced feature set
        # ----------------------------------------------------------
        S = torch.stack(selections, dim=1)  # [B, K, N]
        subset_features = torch.bmm(S, features)  # [B, K, D]

        return subset_features


    def facility_location_select_solution3(
            self,
            features: torch.Tensor,  # [B, N, D]
            Q: torch.Tensor,  # [B, N]
            atten: torch.Tensor | None = None,  # [B, N, N] if sim_from="attention"
            sim_from: str = "cosine",
            K: int | None = None,
            tau_cover: float = 0.3,
            tau_gumbel: float = 0.3,
            eps: float = 1e-6,
    ):
        """
        FACILITY LOCATION (Solution 3)
        --------------------------------------
        Pure Δ-based diversity + soft quality modulation:
            gain(i) = Δ(i) * (1 + η * Q_norm(i))

        - No Top-K
        - No exponent weighting
        - Only ONE selection mechanism (FL)
        - Smooth, stable, numerically safe
        - sim is computed internally as in your original implementation
        """

        B, N, D = features.shape

        if self.training:
            tau_cover = 0.1
            tau_gumbel = 1e-6
            scaler = 1e6
        else:
            tau_cover = 1e-2
            tau_gumbel = 1e-6
            scaler = 1e6

        if K is None:
            K = int(self.selection_count)
        K = min(K, N)

        # ----------------------------------------------------------
        # 1) Compute similarity matrix sim exactly like before
        # ----------------------------------------------------------
        if sim_from == "attention":
            assert atten is not None, "atten must be provided when sim_from='attention'"
            A = atten / (atten.sum(dim=-1, keepdim=True) + eps)  # row normalize
            A = 0.5 * (A + A.transpose(-1, -2))  # symmetrize
            sim = A.clamp(0.0, 1.0)

        else:  # cosine
            X = F.normalize(features, dim=-1, eps=eps)
            sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)
            sim = 0.5 * (sim + 1.0)  # map [-1,1] -> [0,1]
            sim = sim.clamp(0.0, 1.0)

        # ----------------------------------------------------------
        # 2) Normalize Q into [0,1]
        # ----------------------------------------------------------
        Q = Q.squeeze()

        # Q_norm = Q - Q.min(dim=1, keepdim=True).values
        # Q_norm = Q_norm / (Q_norm.max(dim=1, keepdim=True).values + eps)

        # ----------------------------------------------------------
        # 3) Soft quality factor: 1 + η(t) * Q_norm
        # ----------------------------------------------------------
        # Simple epoch-based schedule
        # if self.training:
        #     eta_max = getattr(self, "eta_max", 1.0)
        #     warmup_epochs = getattr(self, "warmup_epochs", 5)
        #     current_epoch = float(getattr(self, "current_epoch", 0))
        #
        #     eta = min(eta_max, current_epoch / warmup_epochs)
        # else:
        #     eta = getattr(self, "eta_max", 1.0)

        # eta = self.q_exp
        # eta = torch.tensor(eta, device=features.device, dtype=features.dtype)

        # multiplier for Δ
        quality_factor = self.q_exp * Q  # [B, N]

        # ----------------------------------------------------------
        # 4) Coverage accumulator C (initially empty)
        # ----------------------------------------------------------
        C = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        chosen_mask = torch.zeros_like(C)
        selections = []

        # ----------------------------------------------------------
        # 5) Seed selection (Δ-like, no Q influence yet)
        # ----------------------------------------------------------
        seed_scores = sim.sum(dim=-1)  # [B, N]
        s = F.gumbel_softmax(seed_scores * scaler, tau=tau_gumbel, hard=True)

        selections.append(s)
        chosen_mask += s

        # update coverage C
        sim_new = torch.einsum("bi,bij->bj", s, sim)
        C = torch.maximum(C, sim_new)

        # ----------------------------------------------------------
        # 6) Greedy Facility Location iterations
        # ----------------------------------------------------------
        for _ in range(1, K):
            # Δ_ij = relu(sim(i,j) - C_j)
            delta = torch.relu(sim - C.unsqueeze(1))  # [B, N, N]

            # Δ(i) = Σ_j Δ_ij
            delta_sum = delta.sum(dim=-1)  # [B, N]

            # Unified single-mechanism gain:
            #     gain(i) = Δ(i) * (1 + η * Q_norm(i))
            gain = delta_sum * quality_factor  # [B, N]

            # mask already chosen tokens
            gain = gain.masked_fill(chosen_mask.bool(), -1e9)

            # soft argmax
            s = F.gumbel_softmax(gain * scaler, tau=tau_gumbel, hard=True)
            selections.append(s)
            chosen_mask += s

            # update coverage after selecting i*
            sim_new = torch.einsum("bi,bij->bj", s, sim)
            C = torch.maximum(C, sim_new)

        # ----------------------------------------------------------
        # 7) Output reduced feature set
        # ----------------------------------------------------------
        S = torch.stack(selections, dim=1)  # [B, K, N]
        subset_features = torch.bmm(S, features)  # [B, K, D]

        return subset_features

    def facility_location_select_fixed_balance_cached_mult_top_K(
            self,
            features: torch.Tensor,  # [B, N, D]
            Q: torch.Tensor,  # [B, N] or [B, N, 1]
            atten: torch.Tensor | None = None,
            sim_from: str = "cosine",
            K: int | None = None,
            lambda_qdiv: float = 0.95,
            tau_cover: float = 0.7,
            tau_gumbel: float = 0.7,
            use_relu_surrogate: bool = True,
            eps: float = 1e-6,
    ):
        """
        Facility Location with exponent-weighted sim and q,
        **plus Top-M candidate filtering before FL**.
        """

        sim_exp = 1.0
        q_exp = 1.0

        # -------------------------
        # Utils
        # -------------------------
        def _to_bn(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(-1) if (x.dim() == 3 and x.size(-1) == 1) else x

        # -------------------------
        # Hyperparameters
        # -------------------------
        if self.training:
            tau_cover = 0.1
            tau_gumbel = 1e-6
            scaler = 1e6
        else:
            tau_cover = 1e-2
            tau_gumbel = 1e-6
            scaler = 1e6

        B, N, D = features.shape
        if K is None:
            K = int(self.selection_count)
        K = min(K, N)

        # ================================================================
        # 1) Compute similarity matrix sim
        # ================================================================
        if sim_from == "attention":
            assert atten is not None, "atten must be provided for sim_from='attention'"
            A = atten / (atten.sum(dim=-1, keepdim=True) + eps)
            sim = 0.5 * (A + A.transpose(-1, -2))
        else:
            X = F.normalize(features, dim=-1, eps=eps)
            sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)
            sim = 0.5 * (sim + 1.0)

        # ================================================================
        # 2) Quality weights q ∈ [0,1]
        # ================================================================
        q = _to_bn(Q).clamp_min(eps)
        q = q / q.sum(dim=1, keepdim=True)  # [B,N]

        # ================================================================
        # 3) NEW: Top-M candidate filtering BEFORE FL
        # ================================================================
        # ratio used: default = 4×K unless user specifies
        # topM_ratio = getattr(self, "topM_ratio", 4.0)
        # M = min(N, int(topM_ratio * K))
        M = K + int((N - K) / 2)
        rawQ = _to_bn(Q)  # [B, N]
        _, top_idx = torch.topk(rawQ, k=M, dim=1)  # [B, M]

        # Gather helpers
        batch = torch.arange(B, device=features.device)[:, None]  # [B,1]

        # Restrict features
        features = features[batch, top_idx]  # [B, M, D]
        q = q[batch, top_idx]  # [B, M]

        # --- Correct double-gather for sim[:, top_idx, top_idx] ---
        # Step 1: gather rows → [B, M, N]
        sim_rows = sim[batch, top_idx]

        # Step 2: gather columns from row-restricted sim → [B, M, M]
        sim = sim_rows[batch, :, top_idx]

        # update N to M
        _, N, _ = features.shape
        K = min(K, N)

        # ================================================================
        # 4) Precompute quality-weighted similarity (exponent version)
        # ================================================================
        #sim_q = (sim ** sim_exp) * (q.unsqueeze(1) ** q_exp)  # [B,N,N]
        sim_q = sim * q.unsqueeze(1)   # [B,N,N]

        # ------------------------------------------------
        # Storage for coverage + mask + selection list
        # ------------------------------------------------
        C = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        chosen_mask = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        selections = []

        # ================================================================
        # 5) Step 1: seed selection (unchanged)
        # ================================================================
        seed_scores = sim_q.sum(dim=-1)  # [B,N]
        s = F.gumbel_softmax(seed_scores * scaler, tau=tau_gumbel, hard=True)
        selections.append(s)
        chosen_mask += s

        sim_new = torch.einsum("bi,bij->bj", s, sim)
        if use_relu_surrogate:
            C = torch.maximum(C, sim_new)
        else:
            C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # ================================================================
        # 6) Step 2..K: FL greedy loop (unchanged except restricted M tokens)
        # ================================================================
        for _ in range(1, K):

            if use_relu_surrogate:
                delta = torch.relu(sim - C.unsqueeze(1))  # [B,N,N]
            else:
                L = (sim - C.unsqueeze(1)) / tau_cover
                delta = tau_cover * F.softplus(L)

            #gain_qual = ((delta ** sim_exp) * (q.unsqueeze(1) ** q_exp)).sum(dim=-1)  # [B,N]
            gain_qual = ((delta ) * (q.unsqueeze(1) )).sum(dim=-1)  # [B,N]

            gain = gain_qual
            gain = gain.masked_fill(chosen_mask.bool(), -1e9)

            s = F.gumbel_softmax(gain * scaler, tau=tau_gumbel, hard=True)
            selections.append(s)
            chosen_mask += s

            sim_new = torch.einsum("bi,bij->bj", s, sim)
            if use_relu_surrogate:
                C = torch.maximum(C, sim_new)
            else:
                C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # ================================================================
        # 7) Output (unchanged)
        # ================================================================
        S = torch.stack(selections, dim=1)  # [B,K,N]
        subset_features = torch.bmm(S, features)  # [B,K,D]
        return subset_features

    def facility_location_select_fixed_balance_cached_mult(
            self,
            features: torch.Tensor,  # [B, N, D]
            Q: torch.Tensor,  # [B, N] or [B, N, 1] (importance weights)
            atten: torch.Tensor | None = None,  # [B, N, N] if sim_from="attention"
            sim_from: str = "cosine",  # "cosine" | "attention"
            K: int | None = None,
            lambda_qdiv: float = 0.95,  # 1.0 = all quality, 0.0 = all diversity
            tau_cover: float = 0.7,  # coverage smooth-max temperature
            tau_gumbel: float = 0.7,  # selection temperature
            use_relu_surrogate: bool = True,  # hinge vs. softplus/LSE surrogate
            eps: float = 1e-6,
    ):
        """Optimized Facility Location with precomputed caches to reduce per-iteration cost."""

        sim_exp = self.sim_exp
        q_exp = self.q_exp
        def _to_bn(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(-1) if (x.dim() == 3 and x.size(-1) == 1) else x

        if self.training:
            tau_cover = 0.1
            tau_gumbel = 1e-6
            scaler = 1e6
        else:
            tau_cover = 1e-2
            tau_gumbel = 1e-6
            scaler = 1e6

        B, N, D = features.shape
        if K is None:
            K = int(self.selection_count)
        K = min(K, N)

        # --- similarity matrix sim ∈ [B,N,N] ---
        if sim_from == "attention":
            assert atten is not None, "atten must be provided when sim_from='attention'"
            A = atten / (atten.sum(dim=-1, keepdim=True) + eps)
            sim = 0.5 * (A + A.transpose(-1, -2))
        else:
            X = F.normalize(features, dim=-1, eps=eps)
            sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)
            sim = 0.5 * (sim + 1.0)

        # --- quality weights ---
        q = _to_bn(Q).clamp_min(eps)
        q = q / q.sum(dim=1, keepdim=True)

        # --- Precompute caches ---
        sim_q = (sim ** sim_exp ) * (q.unsqueeze(1) ** q_exp)  # [B, N, N]  quality-weighted similarities
        #sim_mean = sim.mean(dim=-1)  # [B, N]     average similarity (for gain_div)
        C = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        chosen_mask = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        selections = []

        # ---- Step 1: Seeding with quality coverage ----
        seed_scores = (sim_q).sum(dim=-1)  # Σ_j q_j * sim(i,j)
        s = F.gumbel_softmax(seed_scores * scaler, tau=tau_gumbel, hard=True)
        selections.append(s)
        chosen_mask = chosen_mask + s

        # ---- Coverage update ----
        sim_new = torch.einsum("bi,bij->bj", s, sim)
        if use_relu_surrogate:
            C = torch.maximum(C, sim_new)
        else:
            C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # ---- Step 2..K: Iterative selection ----
        for _ in range(1, K):
            # Compute Δ_ij only once, reusing cached sim
            if use_relu_surrogate:
                delta = torch.relu(sim - C.unsqueeze(1))
            else:
                L = (sim - C.unsqueeze(1)) / tau_cover
                delta = tau_cover * F.softplus(L)

            # Gain computations (using cached sim_q and sim_mean)

            gain_qual = ((delta ** sim_exp) * (q.unsqueeze(1) ** q_exp)).sum(dim=-1)  # [B,N]
            #gain_div = (delta.mean(dim=-1) + sim_mean) / 2.0  # optional regularization
            #gain = lambda_qdiv * gain_qual + (1 - lambda_qdiv) * gain_div
            gain = gain_qual

            gain = gain.masked_fill(chosen_mask.bool(), -1e9)

            s = F.gumbel_softmax(gain * scaler, tau=tau_gumbel, hard=True)
            selections.append(s)
            chosen_mask = chosen_mask + s

            sim_new = torch.einsum("bi,bij->bj", s, sim)
            if use_relu_surrogate:
                C = torch.maximum(C, sim_new)
            else:
                C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # --- Output subset ---
        S = torch.stack(selections, dim=1)  # [B,K,N]
        subset_features = torch.bmm(S, features)  # [B,K,D]
        return subset_features


    def facility_location_select_fixed_balance_cached(
            self,
            features: torch.Tensor,  # [B, N, D]
            Q: torch.Tensor,  # [B, N] or [B, N, 1] (importance weights)
            atten: torch.Tensor | None = None,  # [B, N, N] if sim_from="attention"
            sim_from: str = "cosine",  # "cosine" | "attention"
            K: int | None = None,
            lambda_qdiv: float = 0.95,  # 1.0 = all quality, 0.0 = all diversity
            tau_cover: float = 0.7,  # coverage smooth-max temperature
            tau_gumbel: float = 0.7,  # selection temperature
            use_relu_surrogate: bool = True,  # hinge vs. softplus/LSE surrogate
            eps: float = 1e-6,
    ):
        """Optimized Facility Location with precomputed caches to reduce per-iteration cost."""

        def _to_bn(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(-1) if (x.dim() == 3 and x.size(-1) == 1) else x

        if self.training:
            tau_cover = 0.1
            tau_gumbel = 1e-6
            scaler = 1e6
        else:
            tau_cover = 1e-2
            tau_gumbel = 1e-6
            scaler = 1e6

        B, N, D = features.shape
        if K is None:
            K = int(self.selection_count)
        K = min(K, N)

        # --- similarity matrix sim ∈ [B,N,N] ---
        if sim_from == "attention":
            assert atten is not None, "atten must be provided when sim_from='attention'"
            A = atten / (atten.sum(dim=-1, keepdim=True) + eps)
            sim = 0.5 * (A + A.transpose(-1, -2))
        else:
            X = F.normalize(features, dim=-1, eps=eps)
            sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)
            sim = 0.5 * (sim + 1.0)

        # --- quality weights ---
        q = _to_bn(Q).clamp_min(eps)
        q = q / q.sum(dim=1, keepdim=True)

        # --- Precompute caches ---
        sim_q = sim * q.unsqueeze(1)  # [B, N, N]  quality-weighted similarities
        sim_mean = sim.mean(dim=-1)  # [B, N]     average similarity (for gain_div)
        C = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        chosen_mask = torch.zeros(B, N, device=features.device, dtype=features.dtype)
        selections = []

        # ---- Step 1: Seeding with quality coverage ----
        seed_scores = (sim_q).sum(dim=-1)  # Σ_j q_j * sim(i,j)
        s = F.gumbel_softmax(seed_scores * scaler, tau=tau_gumbel, hard=True)
        selections.append(s)
        chosen_mask = chosen_mask + s

        # ---- Coverage update ----
        sim_new = torch.einsum("bi,bij->bj", s, sim)
        if use_relu_surrogate:
            C = torch.maximum(C, sim_new)
        else:
            C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # ---- Step 2..K: Iterative selection ----
        for _ in range(1, K):
            # Compute Δ_ij only once, reusing cached sim
            if use_relu_surrogate:
                delta = torch.relu(sim - C.unsqueeze(1))
            else:
                L = (sim - C.unsqueeze(1)) / tau_cover
                delta = tau_cover * F.softplus(L)

            # Gain computations (using cached sim_q and sim_mean)
            gain_qual = (delta * q.unsqueeze(1)).sum(dim=-1)  # [B,N]
            gain_div = (delta.mean(dim=-1) + sim_mean) / 2.0  # optional regularization
            gain = lambda_qdiv * gain_qual + (1 - lambda_qdiv) * gain_div

            gain = gain.masked_fill(chosen_mask.bool(), -1e9)

            s = F.gumbel_softmax(gain * scaler, tau=tau_gumbel, hard=True)
            selections.append(s)
            chosen_mask = chosen_mask + s

            sim_new = torch.einsum("bi,bij->bj", s, sim)
            if use_relu_surrogate:
                C = torch.maximum(C, sim_new)
            else:
                C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # --- Output subset ---
        S = torch.stack(selections, dim=1)  # [B,K,N]
        subset_features = torch.bmm(S, features)  # [B,K,D]
        return subset_features

    def facility_location_select(
            self,
            features: torch.Tensor,  # [B, N, D]
            Q: torch.Tensor,  # [B, N] or [B, N, 1]  (importance weights q_j)
            atten: torch.Tensor | None = None,  # [B, N, N] if sim_from="attention"
            sim_from: str = "cosine",  # "cosine" | "attention"
            K: int | None = None,
            tau_gumbel: float = 0.5,  # anneal during training (e.g., 0.7 -> 0.07)
            tau_cover: float = 0.5,
            eps: float = 1e-6,
    ):
        """
        Facility Location (FL) selector (differentiable greedy).

        FL objective (per image):
            F(S) = sum_j q_j * max_{i in S} sim(i, j)

        Soft greedy step:
            Maintain coverage logits C_j ≈ max_i sim(i,j) via:
                C_new = tau * log( exp(C/tau) + exp((s @ sim)/tau) )
            Candidate gain (soft):
                gain_i = sum_j q_j * tau * softplus((sim(i,j) - C_j)/tau)

        Returns:
            subset_features : [B, K, D]
            selections      : [B, K, N]  (ST one-hots per step)
            aux             : dict with diagnostics
        """

        def _to_bn(x: torch.Tensor) -> torch.Tensor:
            # ensure [B, N]
            return x.squeeze(-1) if (x.dim() == 3 and x.size(-1) == 1) else x

        if self.training:
            # tau = 0.5 if (tau is None) else tau
            # scaler = 1.0

            tau_cover = 0.1
            tau = 1e-4
            scaler = 1e6

        else:
            tau_cover = 1e-2
            tau = 1e-6
            scaler = 1e6

        #print('tau_cover = {}, tau = {}, scaler = {}'.format(tau_cover, tau, scaler))
        B, N, D = features.shape
        if K is None:
            K = int(self.selection_count)
        K = min(K, N)
        if sim_from == "attention":
            assert atten is not None, "atten must be provided when sim_from='attention'"
            A = atten / (atten.sum(dim=-1, keepdim=True) + eps)
            sim = 0.5 * (A + A.transpose(-1, -2))  # ~[0,1]
        else:
            X = F.normalize(features, dim=-1, eps=eps)
            sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)
            sim = 0.5 * (sim + 1.0)


        # ---------- 2) Importance weights q_j from Q ----------
        q = _to_bn(Q).clamp_min(eps)  # [B,N]
        q = q / q.sum(dim=1, keepdim=True)  # normalize per image

        # ---------- 3) Greedy FL with soft marginal gains ----------
        # Coverage logits C_j (approx max sim over selected); start very negative.
        C = torch.full((B, N), -1e9, device=features.device, dtype=features.dtype)

        selections = []
        chosen_mask = torch.zeros(B, N, device=features.device, dtype=features.dtype)

        # ---- Iteration 0: seed with score_i = sum_j q_j * sim(i,j) ----
        # correct einsum: seed[b,i] = Σ_j q[b,j] * sim[b,i,j]
        seed_scores = torch.einsum("bj,bij->bi", q, sim)  # [B,N]
        #soft0 = F.gumbel_softmax(seed_scores, tau=tau, hard=False)  # [B,N]
        s = F.gumbel_softmax(seed_scores * scaler, tau=tau, hard=True)  # [B,N]
        #s = hard0 - soft0.detach() + soft0  # ST one-hot
        selections.append(s)
        chosen_mask = chosen_mask + s

        # update coverage: C_new = τ * log( exp(C/τ) + exp((s @ sim)/τ) )
        # s @ sim: [B,N]; einsum "bi,bij->bj"
        sim_new = torch.einsum("bi,bij->bj", s, sim)  # [B,N]
        C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # ---- Subsequent K-1 selections ----
        for _ in range(1, K):
            # L = (sim - C.unsqueeze(1)) / tau  -> [B,N,N]
            L = (sim - C.unsqueeze(1)) / tau_cover
            # gain_i = Σ_j q_j * τ * softplus( L_{i,j} )
            gain = (q.unsqueeze(1) * (tau_cover * F.softplus(L))).sum(dim=-1)  # [B,N]

            # forbid already selected tokens (use large negative, not -inf)
            gain = gain.masked_fill(chosen_mask.bool(), -1e3)

            #soft = F.gumbel_softmax(gain, tau=tau, hard=False)  # [B,N]
            s = F.gumbel_softmax(gain * scaler, tau=tau, hard=True)  # [B,N]
            #s = hard - soft.detach() + soft  # ST one-hot

            selections.append(s)
            chosen_mask = chosen_mask + s

            # update coverage
            sim_new = torch.einsum("bi,bij->bj", s, sim)  # [B,N]
            C = tau_cover * torch.log(torch.exp(C / tau_cover) + torch.exp(sim_new / tau_cover))

        # ---------- 4) Materialize subset ----------
        S = torch.stack(selections, dim=1)  # [B,K,N]
        subset_features = torch.bmm(S, features)  # [B,K,D]

        # aux = {
        #     "q_used": q,  # [B,N]
        #     "sim_mean": sim.mean(dim=(1, 2)),  # [B]
        #     "coverage_mean": C.mean(dim=1),  # [B]
        # }
        return subset_features

    def differentiable_fps_batch_features(
            self,
            features: torch.Tensor,  # [B, N, D]
            Q: torch.Tensor,  # [B, N]  quality per token (any scale)
            atten: torch.Tensor | None = None,  # [B, N, N] optional, only used if self.balancer exists
            cls_logits: torch.Tensor | None = None,  # [B, C] optional, only if self.balancer exists
            D_cos: torch.Tensor | None = None,  # [B, N, N] optional precomputed cosine distance
            tau: float | None = None,  # soft-min temperature; recommend anneal (e.g., 1.0 -> 0.05)
            eps: float = 1e-6,
    ):
        """
        Differentiable FPS with *unary* quality modulation:

          1) First selection: argmax(Q)
          2) Next selections: select j that maximizes  q_gate[j] * min_{i in S} D_cos[i, j]

        where q_gate[j] = softmax(Q[j]) ** beta, normalized to max=1 for stability.
        If a balancer exists, beta is adapted per image; else a fixed beta is used.

        Returns:
            subset_features : [B, K, D]
            selections      : [B, K, N] (one-hot)
            aux             : dict with diagnostics (beta, q_gate, geom)
        """
        B, N, D = features.shape
        K = int(self.selection_count)
        K = min(K, N)  # no padding
        device = features.device
        dtype = features.dtype
        Q = Q.squeeze()
        # ---------- distances: cosine geometry ----------
        if D_cos is None:
            X = F.normalize(features, dim=-1, eps=eps)  # [B,N,D]
            sim = torch.einsum("bid,bjd->bij", X, X).clamp(-1, 1)  # [B,N,N]
            #norm sim to be in [0,1]
            sim = (sim + 1.0) / 2.0
            D_cos = 1.0 - sim  # [B,N,N]

        # ---------- quality gate q_gate (per candidate) ----------
        # Base probabilities from Q
        q_prob = F.softmax(Q, dim=-1).clamp_min(eps)  # [B,N]

        # Decide beta (quality exponent)
        if hasattr(self, "balancer") and isinstance(self.balancer, nn.Module):
            # Build a small feature vector per image (no masking; atten is [B,N,N] if provided)
            feats = []
            # Attention stats (if available)
            if atten is not None:
                A = atten / (atten.sum(dim=-1, keepdim=True) + eps)
                P = A.clamp_min(eps)
                # compact, batched stats
                row_entropy = -(P * P.log()).sum(dim=-1).mean(dim=1)  # [B]
                top2 = P.topk(2, dim=-1).values
                row_margin = (top2[..., 0] - top2[..., 1]).mean(dim=1)  # [B]
                col_std = A.sum(dim=-2).std(dim=1, unbiased=False)  # [B]
                feats += [row_entropy.unsqueeze(-1), row_margin.unsqueeze(-1), col_std.unsqueeze(-1)]
            # Q stats
            probs = q_prob  # already normalized
            q_entropy = -(probs * probs.log()).sum(dim=1) / torch.log(torch.tensor(N, device=device, dtype=dtype))
            q_std = Q.std(dim=1, unbiased=False)
            q_max = Q.max(dim=1).values
            feats += [q_entropy.unsqueeze(-1), q_std.unsqueeze(-1), q_max.unsqueeze(-1)]
            # Optional CLS confidence
            if (cls_logits is not None) and getattr(self, "use_cls_logits", False):
                p_cls = F.softmax(cls_logits, dim=-1).clamp_min(eps)
                top2c = p_cls.topk(2, dim=-1).values
                cls_margin = top2c[:, 0] - top2c[:, 1]
                cls_entropy = -(p_cls * p_cls.log()).sum(dim=1)
                feats += [cls_margin, cls_entropy]
            # K/N context
            k_ratio = torch.full((B,), K / float(N), device=device, dtype=dtype)
            feats += [k_ratio.unsqueeze(-1)]

            # stack safely
            feats = [f if torch.is_tensor(f) else torch.full((B,), float(f), device=device, dtype=dtype) for f in
                     feats]
            x_bal = torch.stack(feats, dim=1).squeeze()  # [B, in_dim]

            # balancer outputs in (0,1); map to beta range
            #beta_min = getattr(self, "q_beta_min", 3.5)
            #beta_max = getattr(self, "q_beta_max", 10.0)
            s = self.balancer(x_bal).clamp(0.0, 1.0).squeeze(1)  # [B]
            #beta = beta_min + (beta_max - beta_min) * s  # [B]
            beta_broadcast = s.view(B, 1)  # [B,1]
        else:
            # fixed beta if no balancer
            beta_val = getattr(self, "q_beta_fixed", 1.0)
            beta_broadcast = torch.full((B, 1), float(beta_val), device=device, dtype=dtype)  # [B,1]

        # q_gate in [0,1], max-normalized for scale stability
        q_gate = (Q ** beta_broadcast).clamp_min(eps)  # [B,N]
        q_gate = q_gate / (q_gate.max(dim=1, keepdim=True).values + eps)  # [B,N]

        # ---------- FPS selection ----------
        # temperature & scaling (use moderate values to keep gradients healthy)
        if self.training:
            # tau = 0.5 if (tau is None) else tau
            # scaler = 1.0
            tau = 1e-6
            scaler = 1e6

        else:
            tau = 1e-6
            scaler = 1e6

        # 1) first token: argmax Q via Gumbel-Top-1
        first = F.gumbel_softmax(q_gate * scaler, tau=tau, hard=True)  # [B,N]
        selections = [first]
        selected = first.unsqueeze(1)  # [B,1,N]
        chosen_mask = first.clone()  # [B,N]

        # 2) subsequent K-1 picks
        for _ in range(1, K):
            # distances from selected set to each candidate (soft-min over selected)
            sel_d = torch.bmm(selected, D_cos)  # [B,S,N]
            d_base = -torch.logsumexp(-sel_d / tau, dim=1) * tau  # [B,N]

            # unary quality modulation on candidates
            d = d_base * (q_gate ** self.gamma) # [B,N]

            # forbid already selected tokens
            d = d.masked_fill(chosen_mask.bool(), -1e9)

            new = F.gumbel_softmax(d * scaler, tau=tau, hard=True)  # [B,N]
            selections.append(new)
            selected = torch.cat([selected, new.unsqueeze(1)], dim=1)  # [B,S+1,N]
            chosen_mask = chosen_mask + new

        S = torch.stack(selections, dim=1)  # [B,K,N]
        subset_features = torch.bmm(S, features)  # [B,K,D]

        # ---------- diagnostics ----------
        # try:
        #     from math import isfinite  # safe if running in restricted env
        #     geom = geometry_stats_from_matrix(D_cos, Q, M=min(8, N))
        # except Exception:
        #     geom = {}
        #
        # aux = {
        #     "beta": beta_broadcast.squeeze(1),  # [B] (per-image quality exponent)
        #     "q_gate": q_gate,  # [B,N] (what modulated candidates)
        #     "geom": geom,
        # }
        return subset_features

    def compute_distance_map(self, attention_heads, Q, gamma):
        """
        Compute a distance map from multi-head attention in PyTorch.

        Parameters:
        - attention_heads: torch.Tensor of shape (B, H, N, N),
          the attention maps for B batches, H heads, and N embeddings.

        Returns:
        - distance_map: torch.Tensor of shape (B, N, N),
          the computed distance maps for each batch.
        """
        # Step 1: Aggregate attention across heads (mean aggregation)
        attention_agg = attention_heads.mean(dim=1)  # Shape: (B, N, N)

        # Step 2: Convert attention to distances
        distance_map = 1 - attention_agg  # Shape: (B, N, N)

        # Step 3: Symmetrize the distance map
        #distance_map = (distance_map + distance_map.transpose(-1, -2)) / 2  # Shape: (B, N, N)

        #Q = Q.unsqueeze(2)

        # Compute A_new:
        #   - A**gamma is computed elementwise over [B, N, N].
        #   - Q_unsqueezed**(1-gamma) is computed elementwise over [B, N, 1] and then broadcast to [B, N, N].
        distance_map = torch.pow(distance_map, 1 - gamma) * torch.pow(Q, gamma)
        distance_map = distance_map.transpose(-1, -2)
        #distance_map = (distance_map + distance_map.transpose(-1, -2)) / 2  # Shape: (B, N, N)
        return distance_map

    def forward_gambel(self, x, quality):
        """
        Forward pass to select features using Gumbel-Softmax for differentiable farthest point selection.

        Args:
            x (torch.Tensor): Input features of shape (batch_size, num_features, feature_dim).
            quality (torch.Tensor): Quality scores for each feature of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Core template of selected features.
        """
        # Determine scaling factor and temperature
        if self.training:
            scaler = 1e4  # Moderate scale to avoid numerical instability
            tau = 1.0  # Near-zero temperature for hard selection
        else:
            scaler = 1e4
            tau = 1e-10  # Near-zero temperature for hard selection

        # Rescale quality scores
        scaled_quality = quality * scaler

        # Step 1: Select the first feature based on quality
        first_selection_mask = torch.nn.functional.gumbel_softmax(
            scaled_quality, tau=tau, hard=True, dim=1
        )
        mask = (first_selection_mask == 0).float()  # Mask for excluding selected items
        quality = quality * mask

        # Compute the core template for the first selection
        core_template = (x.transpose(1, 2) @ first_selection_mask).transpose(1, 2)

        # Compute distances from the selected core template to all features
        dist_core_to_template = self.angular_dist_with_norm(core_template, quality, x)

        # Iteratively select features
        for _ in range(self.selection_count - 1):
            # Step 2: Select the next feature based on distance
            next_selection_mask = torch.nn.functional.gumbel_softmax(
                dist_core_to_template * scaler, tau=tau, hard=True, dim=1
            )
            new_core_item = (x.transpose(1, 2) @ next_selection_mask).transpose(1, 2)

            # Exclude the selected feature
            mask = (next_selection_mask == 0).float()
            quality = quality * mask

            # Update distances using a smooth minimum
            dist_new_to_template = self.angular_dist_with_norm(new_core_item, quality, x)
            dist_core_to_template = torch.min(dist_core_to_template, dist_new_to_template)

            # dist_core_to_template = -torch.log(
            #     torch.exp(-dist_core_to_template) + torch.exp(-dist_new_to_template)
            # )

            # Append the new core item
            core_template = torch.cat([core_template, new_core_item], dim=1)

        return core_template

    # def forward(self, x, quality):
    #
    #
    #     #n_quality = quality * 1000.0
    #     n_quality = quality
    #     if self.training:
    #         #scaler = 1.0
    #         # tau = self.tau.sigmoid()
    #         #tau = 1.0
    #         # tau = 1e-10
    #         scaler = 1000000000000.0
    #         tau = 5.0
    #     else:
    #         scaler = 1000000000000.0
    #         tau = 1e-10
    #
    #     # select the feature with highest quality
    #     max_dist_mask = torch.nn.functional.gumbel_softmax(n_quality * scaler, hard=True, tau=tau,
    #                                                        dim=1)
    #     mask = max_dist_mask.to(torch.int) != 1.0  # Invert the mask to select the desired elements
    #     # Apply the mask to candidate_importance
    #     n_quality = n_quality * mask.float()
    #
    #     core_template = (x.transpose(1, 2) @ max_dist_mask).transpose(1, 2)
    #     core_template.require_grad = True
    #     # After each extraction compute the distance from the coretemplate to the whule template x
    #     dist_core_template_to_template = self.angluar_dist_with_norm(core_template, n_quality, x)
    #     dist_core_template_to_template.require_grad = True
    #
    #     for i in range(self.selection_count - 1):
    #         # Extract index of new point
    #         max_dist_mask = torch.nn.functional.gumbel_softmax(dist_core_template_to_template * scaler, hard=True,
    #                                                            tau=tau,
    #                                                            dim=1)
    #         new_core_item = (x.transpose(1, 2) @ max_dist_mask).transpose(1, 2)
    #
    #         mask = max_dist_mask.to(torch.int) != 1.0  # Invert the mask to select the desired elements
    #         # Apply the mask to candidate_importance
    #         n_quality = n_quality * mask.float()
    #
    #         dist_new_selected_to_template = self.angluar_dist_with_norm(new_core_item, n_quality, x)
    #         dist_core_template_to_template = torch.min(dist_core_template_to_template,
    #                                                    dist_new_selected_to_template)
    #         core_template = torch.cat([core_template, new_core_item], dim=1)
    #
    #     return core_template

class Attention_TopK(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., keep_rate=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)
        self.init_n = 14 * 14

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if torch.any(torch.isnan(attn)) or torch.any(torch.isinf(attn)):
            attn = torch.nan_to_num(attn)
            print("NaN or Inf values found in input tensor")

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.keep_rate < 1:
            left_tokens = int(self.keep_rate * self.init_n)
            if left_tokens == N - 1:
                return x, None, None, None, left_tokens
            assert left_tokens >= 1
            cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]

            # _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            # index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            return x, cls_attn, attn

        return x, None, None

# class Attention_TopK(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., keep_rate=1.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.keep_rate = keep_rate
#         assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)
#         self.init_n = 14*14
#
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         if self.keep_rate < 1:
#             left_tokens = int(self.keep_rate * self.init_n)
#             if left_tokens == N - 1:
#                 return x, None, None, None, left_tokens
#             assert left_tokens >= 1
#             cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
#             cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
#             _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
#             index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]
#
#             return x, index, idx
#
#         return  x, None, None


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class SliceInitializedCrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for query, key, and value
        self.W_q = nn.Linear(query_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(key_value_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(key_value_dim, embed_dim, bias=False)

        # Output projection
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=False)

        self._initialize_slicing()

    def _initialize_slicing(self):
        """
        Initialize W_q, W_k, W_v to act as slicing operators for each head.
        """
        for i, linear_layer in enumerate([self.W_q, self.W_k, self.W_v]):
            weight = torch.zeros_like(linear_layer.weight)
            for head in range(self.num_heads):
                start = head * self.head_dim
                end = (head + 1) * self.head_dim
                weight[start:end, start:end] = torch.eye(self.head_dim)
            linear_layer.weight.data.copy_(weight)

        # Initialize W_out to identity
        self.W_out.weight.data.copy_(torch.eye(self.embed_dim))

    def forward(self, query, key_value):
        """
        Args:
            query: Tensor of shape (batch_size, query_seq_len, query_dim)
            key_value: Tensor of shape (batch_size, key_value_seq_len, key_value_dim)
        """
        batch_size, query_seq_len, _ = query.size()
        _, key_value_seq_len, _ = key_value.size()

        # Project queries, keys, and values
        Q = self.W_q(query).view(batch_size, query_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key_value).view(batch_size, key_value_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(key_value).view(batch_size, key_value_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        # Concatenate heads and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, query_seq_len, self.embed_dim)
        output = self.W_out(output)

        return output



class IdentityInitializedCrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Query comes from query_dim input
        self.W_q = nn.Linear(query_dim, embed_dim, bias=False)
        # Key and Value come from key_value_dim input
        self.W_k = nn.Linear(key_value_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(key_value_dim, embed_dim, bias=False)

        # Output projection
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=False)

        self._initialize_identity()

    def _initialize_identity(self):
        # Initialize W_q, W_k, W_v with identity-like blocks
        for linear_layer in [self.W_q, self.W_k, self.W_v]:
            weight = torch.zeros_like(linear_layer.weight)
            num_features = min(linear_layer.weight.shape[0], linear_layer.weight.shape[1])
            for i in range(self.num_heads):
                start = i * self.head_dim
                end = (i + 1) * self.head_dim
                weight[start:end, :num_features] = torch.eye(self.head_dim, num_features)
            linear_layer.weight.data.copy_(weight)

        # Initialize W_out to identity
        self.W_out.weight.data.copy_(torch.eye(self.embed_dim))

    def forward(self, query, key_value):
        """
        Args:
            query: Tensor of shape (batch_size, query_seq_len, query_dim)
            key_value: Tensor of shape (batch_size, key_value_seq_len, key_value_dim)
        """
        batch_size, query_seq_len, _ = query.size()
        _, key_value_seq_len, _ = key_value.size()

        # Project queries, keys, and values
        Q = self.W_q(query).view(batch_size, query_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key_value).view(batch_size, key_value_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(key_value).view(batch_size, key_value_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        # Concatenate heads and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, query_seq_len, self.embed_dim)
        output = self.W_out(output)

        return output




class MultiHeadAttentionNoVTransform(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        # Learnable parameters for Q and K
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(embed_dim))
            self.q_proj.bias.zero_()
            self.k_proj.weight.copy_(torch.eye(embed_dim))
            self.k_proj.bias.zero_()

        pass
        # Optional final output projection
        #self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V):
        # Batch size, query/key lengths, embedding dimension
        B, Q_len, E = Q.size()
        _, K_len, _ = K.size()

        # Project Q and K
        Q = self.q_proj(Q).view(B, Q_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Q_len, D]
        K = self.k_proj(K).view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K_len, D]

        # Use V directly without transformation
        V = V.view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K_len, D]

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, Q_len, K_len]
        attn_weights = F.softmax(scores, dim=-1)  # Normalize along K_len
        head_outputs = attn_weights @ V  # [B, H, Q_len, D]

        # Concatenate heads
        head_outputs = head_outputs.transpose(1, 2).contiguous().view(B, Q_len, E)  # [B, Q_len, E]

        # Apply final projection
        #output = self.out_proj(head_outputs)
        return head_outputs



class Block_TopK(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_rate=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_TopK(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                   keep_rate=keep_rate)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        init_n = 14 * 14
        self.keep_rate = keep_rate
        output_token_count = int(keep_rate * init_n)
        self.coreset_selection_layer = CoresetSelectionLayer(output_token_count, dim)
        #self.token_selector = BatchedTokenSelector(K=output_token_count)
        #self.mha_k_to_x = MultiHeadAttentionNoVTransform(dim, 1)
        #self.mha_k_to_x = IdentityInitializedMultiHeadAttention(dim, num_heads)
        #self.mha_k_to_x = IdentityInitializedCrossAttention(dim, dim, dim, num_heads)
        self.mha_k_to_x = SliceInitializedCrossAttention(dim, dim, dim, 1)



    def forward(self, x):
        B, N, C = x.shape

        tmp, cls_attn, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(tmp)

        if self.keep_rate < 1.0:

            # B, N, C = x.shape
            # non_cls = x[:, 1:]
            # x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]
            # topk = torch.cat([x[:, 0:1], x_others], dim=1)

            #gil
            # gil
            x_no_cls = x[:,1:]
            #gil
            attn_no_cls = attn.mean(dim=1)[:,1:,1:]
            #attn_no_cls = attn.mean(dim=1)
            #attn_no_cls = attn.max(dim=1).values[:, 1:, 1:]
            #attn =
            #attention_probs = attn / attn.sum(dim=(1, 2), keepdim=True)  # Shape: [B, N, N]
            #coreset = self.coreset_selection_layer.forward_gambel(x_no_cls, cls_attn.unsqueeze(-1))
            # coreset = self.coreset_selection_layer.differentiable_fps_batch_features(x_no_cls, cls_attn.unsqueeze(-1), attn_no_cls)
            # coreset = self.coreset_selection_layer.fps_from_inverse_attention(x_no_cls, cls_attn.unsqueeze(-1),
            #                                                                          attn_no_cls)
            # coreset = self.coreset_selection_layer.facility_location_select(x_no_cls, cls_attn.unsqueeze(-1),
            #                                                                          attn_no_cls)
            # coreset = self.coreset_selection_layer.facility_location_select_fixed_balance_adaptive_gamma(x_no_cls, cls_attn.unsqueeze(-1),
            #                                                                          attn_no_cls)

            coreset = self.coreset_selection_layer.facility_location_select_solution3_centrality_balance_v2(x_no_cls, cls_attn.unsqueeze(-1),
                                                                                     attn_no_cls)



            if torch.any(torch.isnan(coreset)) or torch.any(torch.isinf(coreset)):
                print('coreset is nan')

            #coreset = self.token_selector(x_no_cls, attn[:,:,1:,1:], cls_attn.unsqueeze(-1))
            #coreset = self.coreset_selection_layer(x_no_cls, cls_attn.unsqueeze(-1))
            #coreset = nn.functional.scaled_dot_product_attention(coreset, x_no_cls, x_no_cls)
            coreset = self.mha_k_to_x(coreset, x_no_cls)
            coreset = torch.cat([x[:, 0:1], coreset], dim=1)

            x = coreset
            #x = topk

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        n_tokens = x.shape[1] - 1
        return x, n_tokens, None
        # if index is not None:
        #     return x, n_tokens, idx
        # return x, n_tokens, None


class TopKVisionTransformer(VisionTransformer):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', args=None, dyvit_distillation=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                         num_heads, mlp_ratio, qkv_bias, representation_size, distilled,
                         drop_rate, attn_drop_rate, drop_path_rate, embed_layer, norm_layer,
                         act_layer, weight_init)

        token_ratio = args.keep_rate
        pruning_loc = args.reduction_loc

        if len(token_ratio) == 1:
            token_ratio = [token_ratio[0] ** (idx + 1) for idx in range(len(pruning_loc))]

        assert len(token_ratio) == len(
            pruning_loc), f"Mismatch between the pruning location ({pruning_loc}) and token ratios ({token_ratio})"
        print(token_ratio, pruning_loc)

        token_ratio_full = [1 for _ in range(depth)]
        for idx, loc in enumerate(pruning_loc):
            token_ratio_full[loc] = token_ratio[idx]

        del (self.blocks)
        self.num_patches = self.patch_embed.num_patches
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block_TopK(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                keep_rate=token_ratio_full[i])
            for i in range(depth)])

        self.deit_distillation = distilled

        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio

        self.viz_mode = getattr(args, 'viz_mode', False)

        #self.apply(self._init_weights)

    def get_new_module_names(self):
        return []

    def get_reduction_count(self):
        return self.pruning_loc

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)

        if self.viz_mode:
            decisions = {}
            features = {}

        for i, blk in enumerate(self.blocks):
            x, left_token, sample_idx = blk(x)

            if self.viz_mode and sample_idx is not None:
                decisions[i] = sample_idx.clone().detach().cpu().numpy()
                features[i] = x.clone().detach().cpu().numpy()

        if self.viz_mode and 11 not in features.keys():
            features[i] = x.clone().detach().cpu().numpy()
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)

        if self.training:
            return x
        else:
            if self.viz_mode:
                viz_data = {"Kept_Tokens": decisions, "Features": features}
                return x, viz_data
            else:
                return x


class CoresetVit(VisionTransformer):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', args=None, dyvit_distillation=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                         num_heads, mlp_ratio, qkv_bias, representation_size, distilled,
                         drop_rate, attn_drop_rate, drop_path_rate, embed_layer, norm_layer,
                         act_layer, weight_init)

        token_ratio = args.keep_rate
        pruning_loc = args.reduction_loc

        if len(token_ratio) == 1:
            token_ratio = [token_ratio[0] ** (idx + 1) for idx in range(len(pruning_loc))]

        assert len(token_ratio) == len(
            pruning_loc), f"Mismatch between the pruning location ({pruning_loc}) and token ratios ({token_ratio})"
        print(token_ratio, pruning_loc)

        token_ratio_full = [1 for _ in range(depth)]
        for idx, loc in enumerate(pruning_loc):
            token_ratio_full[loc] = token_ratio[idx]

        del (self.blocks)
        self.num_patches = self.patch_embed.num_patches
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block_TopK(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                keep_rate=token_ratio_full[i])
            for i in range(depth)])

        self.deit_distillation = distilled

        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio

        self.viz_mode = getattr(args, 'viz_mode', False)

        #self.apply(self._init_weights)

    def get_new_module_names(self):
        return []

    def get_reduction_count(self):
        return self.pruning_loc

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)

        if self.viz_mode:
            decisions = {}
            features = {}

        for i, blk in enumerate(self.blocks):
            x, left_token, sample_idx = blk(x)

            if self.viz_mode and sample_idx is not None:
                decisions[i] = sample_idx.clone().detach().cpu().numpy()
                features[i] = x.clone().detach().cpu().numpy()

        if self.viz_mode and 11 not in features.keys():
            features[i] = x.clone().detach().cpu().numpy()
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)

        if self.training:
            return x
        else:
            if self.viz_mode:
                viz_data = {"Kept_Tokens": decisions, "Features": features}
                return x, viz_data
            else:
                return x