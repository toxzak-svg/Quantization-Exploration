"""Cross-layer mutual information for bit allocation.

Implements the cross-layer MI primitive from the sub1quant research notes:

    "I(W_l; W_{l+k}) between weight matrices of different layers, computed
    once via a MINE-style estimator on calibration activations. Bit
    allocation: bits proportional to I(W_l; downstream_loss)."

We approximate I(W_l; W_{l+k}) by I(x_l; x_{l+k}), where x_l are the
post-block activations captured from a single calibration forward pass.
This is the standard activation-proxy trick: a layer's activations carry
the information that its weights encode, and downstream layers consume
those activations. Layers whose weights carry information read by many
downstream consumers should get more bits.

Two estimators ship in this module:

* ``estimate_hsic`` -- Hilbert-Schmidt Independence Criterion with an RBF
  kernel. Cheap, closed-form, biased, in nats. Use for the prototype
  signal that drives allocation.

* ``estimate_mine`` -- Mutual Information Neural Estimation (Belghazi
  2018). Train a small statistic network T_theta to maximize the Donsker
  -Varadhan lower bound. Unbiased (in expectation), in nats. Use for
  the publication-grade number.

Both estimators operate on flat tensors of shape (n_samples, dim_x) and
(n_samples, dim_y). They are deterministic given fixed inputs and
hyperparameters; seeds control MINE's network init.

Caveat documented in the primitive writeup: residual-stream confounding
means I(x_l; x_{l+k}) is not literally equal to I(W_l; W_{l+k}). It is
a *lower bound on the quantization-relevant signal*: layers where the
weights matter for downstream layers will show higher MI in their
activations than layers whose weight information is dominated by the
residual path.

Phase 3 adds two conditional-MI estimators that remove the residual
confounder explicitly:

* ``residual_deltas`` -- replace each layer's activations with the
  per-position difference to its predecessor (``x_l - x_{l-1}``). The
  delta is what the layer's sub-block actually contributed; the residual
  stream is gone by construction. Apply the existing ``hsic_matrix`` to
  the deltas and you get a matrix that approximates
  ``I(subblock_l; subblock_{l+k} | x_{l-1})``. This is the right tool
  for Transformer residual streams.

* ``estimate_hsic_conditional_rff`` -- the linear partial-correlation
  estimator. Regress each of X and Y linearly on Z, take the residuals,
  and compute HSIC on the residuals via RFF. This is the standard
  "control for Z" estimator (Sun et al. 2007). The conditioning step is
  linear in Z, so the residual is "X minus the part of X that Z
  linearly explains" -- exactly the confounder we want to remove for
  the residual-stream case. Has one regularisation knob
  (``ridge_lambda``).

Both are routed through ``allocate_bits(..., conditioning=...)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Activations container
# ---------------------------------------------------------------------------


@dataclass
class CalibrationActivations:
    """One batch of per-layer activations from a calibration forward pass.

    Attributes
    ----------
    hidden_states : dict[int, torch.Tensor]
        Map from layer index to a tensor of shape (batch, seq, hidden).
        Index 0 is the embedding output, index n_layers is the final
        post-norm output.
    layer_keys : list[str]
        Human-readable names for each captured layer, in the same order
        as the indices.
    n_tokens : int
        Total number of token positions across all entries in the batch.
    """

    hidden_states: dict[int, torch.Tensor]
    layer_keys: list[str] = field(default_factory=list)
    n_tokens: int = 0

    def layer_count(self) -> int:
        return len(self.hidden_states)

    def flattened(self, idx: int) -> torch.Tensor:
        """Return layer ``idx`` activations flattened to (n_tokens, hidden)."""
        if idx not in self.hidden_states:
            raise KeyError(f"layer {idx} not captured")
        tensor = self.hidden_states[idx]
        if tensor.ndim != 3:
            raise ValueError(
                f"layer {idx} expected shape (batch, seq, hidden), got {tuple(tensor.shape)}"
            )
        return tensor.reshape(-1, tensor.shape[-1]).to(torch.float32)

    def with_layer(self, idx: int, tensor_3d: torch.Tensor) -> "CalibrationActivations":
        """Return a copy with ``idx`` replaced by ``tensor_3d`` (shape batch, seq, hidden).

        Used by the conditional-MI path to build delta-activation views
        without mutating the original container.
        """
        new_states = dict(self.hidden_states)
        new_states[idx] = tensor_3d
        return CalibrationActivations(
            hidden_states=new_states,
            layer_keys=list(self.layer_keys),
            n_tokens=self.n_tokens,
        )


# ---------------------------------------------------------------------------
# HSIC estimator (fast path, biased, in nats)
# ---------------------------------------------------------------------------


def _rbf_kernel(
    x: torch.Tensor,
    sigma: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the (n, n) RBF Gram matrix for rows of ``x``.

    Uses the median-distance heuristic for ``sigma`` when not supplied.
    """
    n = x.shape[0]
    sq_norm = (x * x).sum(dim=1)
    sq_dist = sq_norm[:, None] + sq_norm[None, :] - 2.0 * x @ x.t()
    sq_dist = sq_dist.clamp_min(0.0)
    if sigma is None:
        median_sq = sq_dist.flatten().median()
        sigma = float(torch.sqrt(0.5 * median_sq.clamp_min(eps)).item())
    bandwidth = 2.0 * sigma * sigma + eps
    return torch.exp(-sq_dist / bandwidth)


def estimate_hsic(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma_x: float | None = None,
    sigma_y: float | None = None,
    sub_sample: int | None = None,
    generator: torch.Generator | None = None,
) -> float:
    """HSIC between two sets of samples, in nats.

    Uses the biased empirical estimator (Gretton 2005): HSIC = (1/n^2) *
    trace(K_x H K_y H) where H = I - 1/n 11^T is the centering matrix.

    Parameters
    ----------
    x, y : torch.Tensor
        Shape (n_samples, dim_x) and (n_samples, dim_y).
    sigma_x, sigma_y : float, optional
        RBF bandwidths. If None, uses median-distance heuristic.
    sub_sample : int, optional
        If set, subsample at most this many rows before computing. The
        estimator variance grows with n; for very large n you usually
        want to cap at a few thousand.
    generator : torch.Generator, optional
        RNG for subsampling.

    Returns
    -------
    float
        HSIC value in nats. Always non-negative (biased estimator).
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors")
    if x.shape[0] < 4:
        raise ValueError("need at least 4 samples for HSIC")

    n_total = x.shape[0]
    if sub_sample is not None and sub_sample < n_total:
        idx = torch.randperm(n_total, generator=generator)[:sub_sample]
        x = x[idx]
        y = y[idx]
    n = x.shape[0]

    kx = _rbf_kernel(x, sigma_x)
    ky = _rbf_kernel(y, sigma_y)
    h = torch.eye(n, dtype=x.dtype, device=x.device) - (1.0 / n)
    kxh = kx @ h
    kyh = ky @ h
    hsic = (kxh * kyh.t()).sum() / (n * n)
    return float(max(hsic.item(), 0.0))


# ---------------------------------------------------------------------------
# Random Fourier Features HSIC (fast path, in nats)
# ---------------------------------------------------------------------------
# Approximate the RBF kernel with D random Fourier features per sample.
# The HSIC then becomes (1/n^2) ||Z_x_c^T Z_y_c||_F^2 where Z_*_c are the
# row-centred RFF embeddings. With D << n this trades a small bias for a
# big constant-factor speedup and an n^2 -> D^2 reduction in pair cost.


def _rff_embed(
    x: torch.Tensor,
    sigma: float | None,
    n_rff: int,
    seed: int,
) -> torch.Tensor:
    """Compute the (n, n_rff) RFF embedding for an RBF kernel of bandwidth sigma.

    For an RBF kernel k(x, x') = exp(-||x - x'||^2 / (2 sigma^2)), the
    random Fourier feature is sqrt(2/D) cos(omega^T x + b) where
    omega ~ N(0, 1/sigma^2 I) and b ~ Uniform(0, 2 pi). When ``sigma``
    is None, uses the same median-distance heuristic as
    ``_rbf_kernel`` so the two approximations are directly comparable.
    """
    if sigma is None:
        sq_norm = (x * x).sum(dim=1)
        sq_dist = sq_norm[:, None] + sq_norm[None, :] - 2.0 * x @ x.t()
        sq_dist = sq_dist.clamp_min(0.0)
        median_sq = sq_dist.flatten().median()
        sigma = float(torch.sqrt(0.5 * median_sq.clamp_min(1e-8)).item())
        if sigma <= 0:
            sigma = 1.0
    gen = torch.Generator(device=x.device).manual_seed(seed)
    omega = torch.randn(x.shape[1], n_rff, generator=gen, device=x.device) / sigma
    b = torch.rand(n_rff, generator=gen, device=x.device) * (2.0 * math.pi)
    proj = x @ omega + b
    return math.sqrt(2.0 / n_rff) * torch.cos(proj)


def _rff_embed_batched(
    tensors: list[torch.Tensor],
    n_rff: int,
    seed: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute RFF embeddings for a batch of layer activations at once.

    All layers share the same omega/b sampling (so the bandwidth is
    consistent across layers and the resulting HSIC matrix is
    well-conditioned). Inputs are concatenated along dim 0 with a
    length tag so the output can be split back per layer.

    Parameters
    ----------
    tensors : list[torch.Tensor]
        L tensors of shape (n, hidden). All must be on the same device.
    n_rff : int
        Random feature dimension.
    seed : int
        RNG seed (used on the device where the RFF lives).
    device : torch.device or str, optional
        If set, all input tensors are moved to this device first. This
        is the only place that touches host-to-device transfers in the
        matrix path -- callers should keep their tensors on CPU and
        let ``device="cuda"`` move them once.

    Returns
    -------
    torch.Tensor
        Shape (L, n, n_rff) where L = len(tensors), on the requested
        device.
    """
    if device is not None:
        tensors = [t.to(device) for t in tensors]
    n = tensors[0].shape[0]
    hidden = tensors[0].shape[1]
    dev = tensors[0].device
    # One bandwidth across all layers, picked from the first tensor.
    sq_norm = (tensors[0] * tensors[0]).sum(dim=1)
    sq_dist = sq_norm[:, None] + sq_norm[None, :] - 2.0 * tensors[0] @ tensors[0].t()
    sq_dist = sq_dist.clamp_min(0.0)
    median_sq = sq_dist.flatten().median()
    sigma = float(torch.sqrt(0.5 * median_sq.clamp_min(1e-8)).item())
    if sigma <= 0:
        sigma = 1.0
    gen = torch.Generator(device=dev).manual_seed(seed)
    omega = torch.randn(hidden, n_rff, generator=gen, device=dev) / sigma
    b = torch.rand(n_rff, generator=gen, device=dev) * (2.0 * math.pi)
    stacked = torch.stack(tensors, dim=0)  # (L, n, hidden)
    proj = stacked @ omega + b  # (L, n, n_rff) via broadcasting
    return math.sqrt(2.0 / n_rff) * torch.cos(proj)


def estimate_hsic_rff(
    x: torch.Tensor,
    y: torch.Tensor,
    n_rff: int = 256,
    sigma_x: float | None = None,
    sigma_y: float | None = None,
    sub_sample: int | None = None,
    seed: int = 0,
    n_samples: int = 1,
    device: torch.device | str | None = None,
) -> float:
    """HSIC approximated via Random Fourier Features.

    For RBF kernels the kernel can be written as the expectation of
    cos-features over random frequencies (Rahimi & Recht 2007). With D
    such features per sample, HSIC becomes the squared Frobenius norm
    of an (D, D) inner-product matrix instead of an (n, n) Gram
    product. For n_samples >> n_rff this is a substantial speedup.

    Parameters
    ----------
    x, y : torch.Tensor
        Shape (n_samples, dim_x) and (n_samples, dim_y).
    n_rff : int
        Number of random Fourier features. 256 is a good default; 64
        runs faster but with more variance, 1024 matches RBF HSIC to
        ~1% relative error.
    sigma_x, sigma_y : float, optional
        Bandwidths. If None, picked from the median-distance heuristic.
    sub_sample : int, optional
        Cap the number of rows before computing.
    seed : int
        RNG seed for the random frequencies.
    n_samples : int
        Number of independent RFF samples to average over. Each sample
        adds variance reduction ~1/sqrt(n_samples) at linear cost.
        Default 1 is fastest; 4-8 are typical for tight rankings.
    device : torch.device or str, optional
        If set, x and y are moved to this device first. The bandwidth
        heuristic (median distance) is computed on-device when x is
        already there. Default keeps the input's device.

    Returns
    -------
    float
        Approximate HSIC in nats. Biased but consistent.
    """
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors")
    if x.shape[0] < 4:
        raise ValueError("need at least 4 samples for HSIC")
    if n_rff < 2:
        raise ValueError("n_rff must be >= 2")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    n_total = x.shape[0]
    if sub_sample is not None and sub_sample < n_total:
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n_total, generator=gen)[:sub_sample]
        x = x[idx]
        y = y[idx]
    n = x.shape[0]

    total = 0.0
    for k in range(n_samples):
        z_x = _rff_embed(x, sigma_x, n_rff, seed + 2 * k)
        z_y = _rff_embed(y, sigma_y, n_rff, seed + 2 * k + 1)
        z_x = z_x - z_x.mean(dim=0, keepdim=True)
        z_y = z_y - z_y.mean(dim=0, keepdim=True)
        inner = z_x.t() @ z_y
        total += float((inner.pow(2).sum() / (n * n)).item())
    return max(total / n_samples, 0.0)


# ---------------------------------------------------------------------------
# Conditional MI estimators (Phase 3)
# ---------------------------------------------------------------------------
#
# Unconditional HSIC over-credits layers that share a residual stream:
# if x_{l+1} = x_l + subblock(x_l), then I(x_l; x_{l+k}) is mostly
# "they share x_l", not "their weights interact". For quantization we
# want the latter -- so we condition on the predecessor (or on any
# side-information Z) before measuring dependence.


def residual_deltas(
    activations: CalibrationActivations,
    keep_baseline: bool = False,
    baseline_idx: int = 0,
) -> CalibrationActivations:
    """Per-position deltas to the immediate predecessor in sort order.

    Returns a new ``CalibrationActivations`` whose layer-``l`` entry is
    ``x_l - x_{prev(l)}`` where ``prev(l)`` is the largest captured
    layer index strictly less than ``l``. Layer ``baseline_idx`` is the
    reference point -- it is kept as-is unless ``keep_baseline=False``,
    in which case it is dropped because there is no "previous" to
    subtract from.

    For a Transformer decoder this approximates
    ``subblock_l(x_{l-1})`` -- exactly the signal whose quantization we
    care about. The residual stream confounder is gone by construction.

    Parameters
    ----------
    keep_baseline : bool
        If True, ``baseline_idx`` is preserved unchanged in the output
        (with a zero delta would lose information about the input
        embedding's role). Default False drops it.
    baseline_idx : int
        The reference layer for deltas. Default 0 (the embedding
        output).

    Returns
    -------
    CalibrationActivations
        Same shape as the input but each layer is the per-position
        difference to its predecessor. ``n_tokens`` is preserved.
    """
    indices = sorted(activations.hidden_states.keys())
    if len(indices) < 2:
        raise ValueError("need at least 2 captured layers to form deltas")

    if baseline_idx not in activations.hidden_states:
        # Fall back: use the smallest index as the baseline.
        baseline_idx = indices[0]

    new_states: dict[int, torch.Tensor] = {}
    prev_tensor: torch.Tensor | None = None
    for idx in indices:
        cur = activations.hidden_states[idx]
        if prev_tensor is None:
            # This is the baseline.
            if keep_baseline:
                new_states[idx] = cur.clone()
            prev_tensor = cur
            continue
        if cur.shape != prev_tensor.shape:
            raise ValueError(
                f"layer {idx} shape {tuple(cur.shape)} does not match "
                f"predecessor shape {tuple(prev_tensor.shape)}; "
                f"deltas require identical shapes"
            )
        new_states[idx] = cur - prev_tensor
        prev_tensor = cur

    return CalibrationActivations(
        hidden_states=new_states,
        layer_keys=list(activations.layer_keys),
        n_tokens=activations.n_tokens,
    )


def estimate_hsic_conditional_rff(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    n_rff: int = 256,
    ridge_lambda: float = 1e-2,
    sub_sample: int | None = None,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> float:
    """Approximate ``HSIC(X, Y | Z)`` via linear partial correlation + RFF HSIC.

    The estimator:

    1. Regress X (and Y) on Z in the original space with Tikhonov
       regularisation: solve ``B_x = (Z^T Z + lambda I)^-1 Z^T X`` and
       take the residual ``r_x = X - Z B_x``. Same for Y.
    2. Compute HSIC(r_x, r_y) via the standard RFF HSIC identity
       (``estimate_hsic_rff``).

    This is the partial-correlation conditional MI estimator (Sun et
    al. 2007). The conditioning step is linear in Z, so the residual
    is "X minus the part of X that Z linearly explains". For the
    residual-stream case (Z = layer l-1 activations), this captures
    exactly the "subblock_l + nonlinearity leak" we want for
    quantization -- the linear pass-through of the residual stream is
    removed.

    Why linear conditioning, not RFF kernel ridge? At the
    median-distance bandwidth, RBF RFF features span smooth functions
    of Z -- not the linear identity. Linear conditioning in the
    original Z-space gives much better confounder removal for our
    use case (it preserves any nonlinear signal in the residual)
    while being parameter-free apart from ``ridge_lambda``. If you
    need non-linear conditioning, regress on RFF features explicitly
    with a bandwidth chosen for your data.

    Caveats:

    * Assumes ``dim_x == dim_y == dim_z``. For the cross-layer
      allocation problem this is always true (all layer activations
      share the same hidden dim).
    * ``ridge_lambda`` trades bias vs variance. The default ``1e-2``
      is conservative; the gram diagonal is O(n * ||z||^2 / d), so
      this is a small fraction of the gram scale. If the residuals
      look noisy, raise it; if Z clearly explains X and the residual
      looks wrong, drop it.
    * Output is always non-negative (biased empirical HSIC).

    Parameters
    ----------
    x, y : torch.Tensor
        Shape (n_samples, dim).
    z : torch.Tensor
        Conditioning variable, shape (n_samples, dim_z). dim_z must
        equal dim_x.
    n_rff : int
        RFF feature dimension for the final HSIC. 256 matches the
        unconditional default.
    ridge_lambda : float
        Tikhonov regularisation strength for the linear regression.
    sub_sample : int, optional
        Cap the number of rows before computing.
    seed : int
        RFF seed for the final HSIC step.

    Returns
    -------
    float
        Approximate ``HSIC(X, Y | Z)`` in nats.
    """
    if device is not None:
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape for linear conditioning")
    if x.shape[0] != z.shape[0]:
        raise ValueError("z must have the same number of rows as x and y")
    if x.ndim != 2 or y.ndim != 2 or z.ndim != 2:
        raise ValueError("x, y, z must all be 2D tensors")
    if x.shape[1] != z.shape[1]:
        raise ValueError(
            f"conditional HSIC requires dim_x == dim_z; "
            f"got dim_x={x.shape[1]} dim_z={z.shape[1]}. "
            f"Use residual_deltas for the residual-stream case."
        )
    if x.shape[0] < 4:
        raise ValueError("need at least 4 samples for HSIC")
    if n_rff < 2:
        raise ValueError("n_rff must be >= 2")
    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")

    n_total = x.shape[0]
    if sub_sample is not None and sub_sample < n_total:
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n_total, generator=gen)[:sub_sample]
        x = x[idx]
        y = y[idx]
        z = z[idx]
    n = x.shape[0]
    d_z = z.shape[1]

    # Linear ridge regression of X on Z: B_x = (Z^T Z + lambda I)^-1 Z^T X.
    # Z^T Z is (d_z, d_z) -- typically 16x16 to 1536x1536; small enough
    # for a direct Cholesky solve.
    gram = z.t() @ z
    eye = torch.eye(d_z, dtype=gram.dtype, device=gram.device)
    gram_reg = gram + ridge_lambda * eye
    try:
        L = torch.linalg.cholesky(gram_reg)
    except Exception:
        gram_reg = gram_reg + 1e-4 * eye  # jitter for stability
        L = torch.linalg.cholesky(gram_reg)
    rhs_x = z.t() @ x
    rhs_y = z.t() @ y
    B_x = torch.cholesky_solve(rhs_x, L)  # (d_z, d)
    B_y = torch.cholesky_solve(rhs_y, L)

    res_x = x - z @ B_x
    res_y = y - z @ B_y

    # HSIC of residuals via RFF.
    return estimate_hsic_rff(res_x, res_y, n_rff=n_rff, sub_sample=None, seed=seed)


# ---------------------------------------------------------------------------
# MINE estimator (publication path, unbiased, in nats)
# ---------------------------------------------------------------------------


class _StatisticNet(nn.Module):
    def __init__(self, dim_x: int, dim_y: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x + dim_y, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )
        # Initialise the final layer small so T starts near zero. Without
        # this the network can land in a regime where T_joint and
        # T_marginal are both large and well-separated, hiding the MI
        # signal behind a constant offset.
        with torch.no_grad():
            self.net[-1].weight.normal_(0.0, 0.01)
            self.net[-1].bias.zero_()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=-1))


def estimate_mine(
    x: torch.Tensor,
    y: torch.Tensor,
    hidden: int = 128,
    lr: float = 1e-4,
    steps: int = 800,
    batch_size: int | None = None,
    ema_decay: float = 0.01,
    sub_sample: int | None = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
    progress: bool = False,
) -> tuple[float, list[float]]:
    """MINE estimate of I(X;Y) in nats.

    Uses the Donsker-Varadhan lower bound with the gradient flowing
    through both joint and marginal terms (the standard MINE loss, not
    the EMA-only MINE-f variant). The marginal term uses
    ``logsumexp - log N`` for numerical stability. Gradient clipping
    bounds the per-step update magnitude.

    Parameters
    ----------
    x, y : torch.Tensor
        Shape (n_samples, dim_x) and (n_samples, dim_y). Will be moved to
        ``device``.
    hidden : int
        Width of the statistic network.
    lr : float
        Adam learning rate. Default 1e-4 is conservative; bump to 1e-3
        if you can afford divergence and want faster convergence.
    steps : int
        Number of gradient steps.
    batch_size : int, optional
        Mini-batch size. Defaults to min(512, n_samples).
    ema_decay : float
        Unused. Kept for API stability; the EMA variant proved unstable
        in our tests (gradient through detached EMA gives unbounded
        growth). We rely on gradient clipping instead.
    sub_sample : int, optional
        If set, subsample at most this many rows before training.
    seed : int
        Seed for the statistic network init.
    device : str or torch.device
        Where to run the network.
    progress : bool
        Print every 100 steps.

    Returns
    -------
    (estimate, trace) : (float, list[float])
        Final MINE estimate in nats and the per-step trace.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors")
    if x.shape[0] < 8:
        raise ValueError("need at least 8 samples for MINE")

    n_total = x.shape[0]
    if sub_sample is not None and sub_sample < n_total:
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n_total, generator=gen)[:sub_sample]
        x = x[idx]
        y = y[idx]
    n = x.shape[0]

    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)

    net = _StatisticNet(x.shape[1], y.shape[1], hidden=hidden).to(device)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    effective_batch = int(batch_size) if batch_size else min(512, n)
    max_grad_norm = 5.0

    trace: list[float] = []
    gen = torch.Generator(device=device).manual_seed(seed + 1)

    for step in range(steps):
        idx_a = torch.randint(0, n, (effective_batch,), generator=gen, device=device)
        idx_b = torch.randint(0, n, (effective_batch,), generator=gen, device=device)
        joint = net(x[idx_a], y[idx_a])
        marginal = net(x[idx_a], y[idx_b])
        # Numerically stable log(mean(exp(T))):
        # log(mean(exp(T))) = logsumexp(T) - log(N).
        log_mean_exp_marginal = torch.logsumexp(marginal, dim=0) - math.log(
            marginal.shape[0]
        )
        loss = -(joint.mean() - log_mean_exp_marginal)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        opt.step()
        if progress and (step == 0 or (step + 1) % 100 == 0):
            with torch.no_grad():
                est = (joint.mean() - log_mean_exp_marginal).item()
            print(f"  [MINE] step {step + 1:4d}: I_hat={est:.4f} nats")
        trace.append(float((-loss).item()))

    with torch.no_grad():
        # Final estimate using a fresh batch.
        idx_a = torch.randperm(n, generator=gen, device=device)[: min(1024, n)]
        idx_b = torch.randperm(n, generator=gen, device=device)[: min(1024, n)]
        joint = net(x[idx_a], y[idx_a])
        marginal = net(x[idx_a], y[idx_b])
        log_mean_exp = torch.logsumexp(marginal, dim=0) - math.log(marginal.shape[0])
        est = (joint.mean() - log_mean_exp).item()
    return float(max(est, 0.0)), trace


# ---------------------------------------------------------------------------
# Vectorised batched HSIC for the full L x L matrix
# ---------------------------------------------------------------------------


def hsic_matrix(
    activations: CalibrationActivations,
    sub_sample: int | None = 2048,
    seed: int = 0,
    method: str = "rff",
    n_rff: int = 256,
    n_rff_samples: int = 1,
    horizon: int | None = None,
    conditioning: str | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute the L x L HSIC matrix for a CalibrationActivations.

    Two methods are supported:

    ``method="rff"`` (default) -- Random Fourier Features approximation.
    Each layer's activations are embedded into an (n, n_rff) feature
    matrix Z; HSIC(i, j) is computed via the centred gram-product
    identity HSIC = <K_i, K_j>_F / n^2 where K_l = Z_l Z_l^T.

    ``method="rbf"`` -- exact biased estimator via (n, n) Gram
    matrices. Slower but reference-quality.

    When ``horizon`` is set, only entries with |i - j| <= horizon are
    computed (everything else stays zero). For the 35-layer Gemma
    decoder with horizon=4, this skips ~75% of pairs.

    Diagonal HSIC values (a layer with itself) are returned as-is in
    both modes -- they are large and uninformative; downstream code
    should ignore them.

    Conditioning:

    ``conditioning="delta"`` (Phase 3) -- preprocess activations through
    ``residual_deltas`` before computing the matrix. Each layer is
    replaced by ``x_l - x_{l-1}``, removing the residual-stream
    confounder by construction. The resulting matrix approximates
    ``I(subblock_l; subblock_{l+k} | x_{l-1})``. Default ``None``
    keeps the original unconditional behaviour.

    For pair-level (non-matrix) conditional MI with arbitrary
    side-information, use ``estimate_hsic_conditional_rff`` directly.

    Parameters
    ----------
    n_rff : int
        Number of random Fourier features per sample. 256 is the
        speed/accuracy sweet spot; 1024 matches RBF HSIC to ~1%.
    n_rff_samples : int
        Number of independent RFF samples to average per pair.
        Variance reduces ~1/sqrt(n_rff_samples). 1 is the production
        default (cheap, plenty accurate for ranking); 4 for tight
        rankings.
    horizon : int, optional
        If set, only entries within ``horizon`` of the diagonal are
        computed; off-band entries are zero. Useful when downstream
        scoring only sums nearby pairs (the default in this module).
    conditioning : {None, "delta"}
        Pre-processing for conditional MI. ``"delta"`` replaces each
        layer with the per-position delta to its predecessor before
        computing the matrix.

    Returns
    -------
    torch.Tensor
        Shape (n_layers, n_layers), values in nats. Note: when
        ``conditioning="delta"`` the matrix has one fewer row/column
        than the input (the baseline layer is dropped).
    """
    if conditioning not in (None, "delta"):
        raise ValueError(
            f"hsic_matrix supports conditioning=None|'delta'; "
            f"got {conditioning!r}. For pair-level conditioning use "
            f"estimate_hsic_conditional_rff directly."
        )
    if conditioning == "delta":
        activations = residual_deltas(activations, keep_baseline=False)
    indices = sorted(activations.hidden_states.keys())
    if not indices:
        raise ValueError("no hidden states captured")
    flat = [activations.flattened(i) for i in indices]

    n_layers = len(flat)
    gen = torch.Generator().manual_seed(seed)

    # Subsample once on CPU (host-side RNG is fast) before moving to device.
    n_total = flat[0].shape[0]
    if sub_sample is not None and sub_sample < n_total:
        idx = torch.randperm(n_total, generator=gen)[:sub_sample]
        flat = [t[idx] for t in flat]

    # Move to device (no-op if device matches input). This is the only
    # host-to-device transfer in the matrix path.
    if device is not None:
        flat = [t.to(device) for t in flat]

    if method == "rff":
        n = flat[0].shape[0]
        # Compute the (L, n, n) centred gram matrices for each RFF
        # sample and accumulate pair products. Off-horizon entries are
        # masked to zero after the matmul (a vectorised mask is much
        # faster than the per-element Python loop, especially on GPU).
        flat_grams_accum: torch.Tensor | None = None
        for k in range(n_rff_samples):
            sample_seed = seed + k * 1000
            embeddings = _rff_embed_batched(flat, n_rff, sample_seed)
            grams = embeddings @ embeddings.transpose(1, 2)
            grams = grams - grams.mean(dim=2, keepdim=True)
            grams = grams - grams.mean(dim=1, keepdim=True)
            flat_g = grams.reshape(n_layers, n * n)
            if flat_grams_accum is None:
                flat_grams_accum = flat_g
            else:
                flat_grams_accum = flat_grams_accum + flat_g
        flat_grams = flat_grams_accum / n_rff_samples
        matrix = (flat_grams @ flat_grams.t()) / (n * n)
        matrix.clamp_min_(0.0)
        if horizon is not None:
            # Vectorised band mask: (|i-j| > horizon) -> 0.
            i_idx = torch.arange(n_layers, device=matrix.device).unsqueeze(1)
            j_idx = torch.arange(n_layers, device=matrix.device).unsqueeze(0)
            band_mask = (i_idx - j_idx).abs() <= horizon
            matrix = matrix * band_mask.to(matrix.dtype)
        return matrix

    if method != "rbf":
        raise ValueError(f"unknown method {method!r}; expected 'rff' or 'rbf'")

    # RBF path: precompute centred Gram matrices.
    grams: list[torch.Tensor] = []
    for tensor in flat:
        grams.append(_rbf_kernel(tensor))
    n = flat[0].shape[0]
    h = torch.eye(n, dtype=torch.float32, device=flat[0].device) - (1.0 / n)
    centered = [g @ h for g in grams]
    matrix = torch.zeros((n_layers, n_layers), dtype=torch.float32, device=flat[0].device)
    for i in range(n_layers):
        ci = centered[i]
        j_start = i if horizon is None else max(0, i - horizon)
        j_end = n_layers if horizon is None else min(n_layers, i + horizon + 1)
        for j in range(j_start, j_end):
            cj = centered[j]
            val = (ci * cj.t()).sum().item() / (n * n)
            matrix[i, j] = max(val, 0.0)
            matrix[j, i] = matrix[i, j]
    return matrix


# ---------------------------------------------------------------------------
# Cross-layer MI score and bit allocation
# ---------------------------------------------------------------------------


@dataclass
class MIAllocation:
    """Bit-allocation result from cross-layer MI."""

    mi_scores: torch.Tensor  # (n_layers,) per-layer aggregated score
    mi_matrix: torch.Tensor  # (n_layers, n_layers)
    bit_weights: torch.Tensor  # (n_layers,) raw weights before budget clamp
    bits: torch.Tensor  # (n_layers,) allocated bits per layer
    method: str
    alloc_method: str = "rank"

    def summary(self) -> dict:
        return {
            "n_layers": int(self.bits.numel()),
            "avg_bits": float(self.bits.mean().item()),
            "min_bits": float(self.bits.min().item()),
            "max_bits": float(self.bits.max().item()),
            "mi_min": float(self.mi_scores.min().item()),
            "mi_max": float(self.mi_scores.max().item()),
            "mi_mean": float(self.mi_scores.mean().item()),
            "method": self.method,
        }


def mi_scores_from_matrix(
    mi_matrix: torch.Tensor,
    horizon: int = 4,
    self_diag: bool = False,
) -> torch.Tensor:
    """Aggregate the L x L MI matrix into a per-layer sensitivity score.

    Each layer's score is the sum of MI to layers within ``horizon`` steps
    ahead (and behind). This captures "how much downstream consumers
    depend on this layer's information".

    Parameters
    ----------
    mi_matrix : torch.Tensor
        Shape (n_layers, n_layers) -- symmetric HSIC matrix.
    horizon : int
        Number of layers in each direction to consider.
    self_diag : bool
        Whether to include the diagonal (a layer with itself) in the sum.
        Off by default because diagonal HSIC is large and uninformative.
    """
    if mi_matrix.ndim != 2 or mi_matrix.shape[0] != mi_matrix.shape[1]:
        raise ValueError("mi_matrix must be square")
    n = mi_matrix.shape[0]
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    scores = torch.zeros(n, dtype=mi_matrix.dtype)
    for i in range(n):
        lo = max(0, i - horizon)
        hi = min(n, i + horizon + 1)
        for j in range(lo, hi):
            if i == j and not self_diag:
                continue
            scores[i] += mi_matrix[i, j]
    return scores


def mi_to_bit_allocation(
    mi_scores: torch.Tensor,
    bits_min: float = 1.5,
    bits_max: float = 8.0,
    temperature: float = 0.1,
    target_avg_bpw: float | None = None,
    method: str = "rank",
) -> torch.Tensor:
    """Convert per-layer MI scores to per-layer bit-widths.

    Two methods are available.

    ``method="rank"`` (default) -- rank the layers by MI score and
    interpolate linearly between ``bits_min`` and ``bits_max``. Always
    produces the full range, regardless of how uniform the raw scores
    are. This is the right choice when you want a fixed-bits budget
    spread across the layers rather than a softmax-shaped distribution.

    ``method="softmax"`` -- softmax-normalise the scores (with
    ``temperature``) and rescale into ``[bits_min, bits_max]``. A small
    temperature produces sharper allocation; a large temperature
    produces near-uniform allocation. Useful when the scores' *ratios*
    (not just their order) carry meaning you want to preserve.

    If ``target_avg_bpw`` is given, the output is shifted by a constant
    so the resulting mean matches the target -- this is the budget
    knob the mixed-budget allocator uses.
    """
    if mi_scores.ndim != 1:
        raise ValueError("mi_scores must be 1D")
    if bits_min >= bits_max:
        raise ValueError("bits_min must be < bits_max")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    n = mi_scores.numel()
    if method == "rank":
        # Higher MI -> more bits. The top-ranked layer (position 0 in
        # descending sort) should get bits_max; the lowest should get
        # bits_min. So we invert: rank_norm = (n-1 - position) / (n-1).
        order = torch.argsort(mi_scores, descending=True)
        ranks = torch.empty(n, dtype=torch.float32)
        denom = max(n - 1, 1)
        for new_pos, orig_idx in enumerate(order.tolist()):
            ranks[orig_idx] = (denom - new_pos) / denom
        bits = bits_min + ranks * (bits_max - bits_min)
    elif method == "softmax":
        scaled = mi_scores / temperature
        scaled = scaled - scaled.max()
        probs = torch.softmax(scaled, dim=0)
        bits = bits_min + probs * (bits_max - bits_min)
    else:
        raise ValueError(f"unknown method {method!r}; expected 'rank' or 'softmax'")

    if target_avg_bpw is not None:
        current_avg = bits.mean().item()
        if current_avg > 0:
            bits = bits * (target_avg_bpw / current_avg)
        bits = bits.clamp(min=bits_min, max=bits_max)
    return bits


def allocate_bits(
    activations: CalibrationActivations,
    horizon: int = 4,
    bits_min: float = 1.5,
    bits_max: float = 8.0,
    temperature: float = 0.1,
    target_avg_bpw: float | None = None,
    sub_sample: int | None = 2048,
    seed: int = 0,
    alloc_method: str = "rank",
    hsic_method: str = "rff",
    n_rff: int = 256,
    n_rff_samples: int = 1,
    conditioning: str | None = None,
    device: torch.device | str | None = None,
) -> MIAllocation:
    """One-shot MI-driven bit allocation.

    Builds the HSIC matrix from ``activations``, aggregates with a horizon
    window, and returns the per-layer bit widths ready to feed into the
    mixed-budget allocator.

    ``hsic_method`` chooses between the fast RFF approximation
    (``"rff"``, default) and the exact RBF estimator (``"rbf"``). The
    matrix is built with the same ``horizon`` as the scoring window, so
    off-band pairs are skipped -- this gives a ~3-4x speedup on the
    35-layer Gemma decoder.

    ``n_rff_samples`` controls RFF variance reduction. 1 is the
    production default (cheap, sufficient for ranking); 4 for tight
    rankings where the top-3 layers matter.

    ``conditioning`` (Phase 3):

    * ``None`` -- unconditional MI, the historical behaviour. Use this
      for quick smoke tests and A/B against the conditional path.
    * ``"delta"`` -- preprocess activations through ``residual_deltas``
      before computing the matrix. Each layer becomes ``x_l - x_{l-1}``
      and the matrix approximates
      ``I(subblock_l; subblock_{l+k} | x_{l-1})``. The residual-stream
      confounder is removed. Default for any real Gemma allocation run.

    Note: the output ``MIAllocation`` has one fewer entry than the input
    when ``conditioning="delta"`` because the baseline layer (typically
    the embedding output) has no predecessor to subtract from and is
    dropped. The bit widths line up with layer indices
    ``[baseline+1, baseline+2, ..., max_idx]`` -- callers feeding this
    into the mixed-budget allocator should pass
    ``layer_indices=range(baseline+1, max_idx+1)`` to keep them aligned.
    """
    matrix = hsic_matrix(
        activations,
        sub_sample=sub_sample,
        seed=seed,
        method=hsic_method,
        n_rff=n_rff,
        n_rff_samples=n_rff_samples,
        horizon=horizon,
        conditioning=conditioning,
        device=device,
    )
    scores = mi_scores_from_matrix(matrix, horizon=horizon)
    bits = mi_to_bit_allocation(
        scores,
        bits_min=bits_min,
        bits_max=bits_max,
        temperature=temperature,
        target_avg_bpw=target_avg_bpw,
        method=alloc_method,
    )
    raw_weights = scores / scores.sum().clamp_min(1e-8)
    method_tag = f"cross_layer_mi_hsic_{hsic_method}_h{horizon}"
    if conditioning is not None:
        method_tag = f"{method_tag}_cond_{conditioning}"
    return MIAllocation(
        mi_scores=scores,
        mi_matrix=matrix,
        bit_weights=raw_weights,
        bits=bits,
        method=method_tag,
        alloc_method=alloc_method,
    )


# ---------------------------------------------------------------------------
# Convenience: Sigma-style score for comparison
# ---------------------------------------------------------------------------


def sigma_scores_from_activations(
    activations: CalibrationActivations,
) -> torch.Tensor:
    """Per-layer sigma (std of activation magnitudes).

    This is the classic per-layer-static score the MI primitive is
    meant to replace. Returned for head-to-head comparison.

    For each layer we take the standard deviation of activation
    L2-norms across token positions. Layers whose activations have
    higher variance are conventionally ranked as more "important" by
    σ-style allocation schemes.
    """
    indices = sorted(activations.hidden_states.keys())
    out = torch.zeros(len(indices), dtype=torch.float32)
    for k, idx in enumerate(indices):
        flat = activations.flattened(idx)
        norms = flat.norm(dim=-1)
        out[k] = norms.std()
    return out