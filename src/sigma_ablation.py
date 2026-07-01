from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SIGMAConfig:
    block_size: int = 64
    n_bits: int = 8
    n_taus: int = 8
    rank: int = 32


@dataclass(frozen=True)
class BlockStats:
    n_blocks: int
    block_size: int
    total_elements: int


def make_blocks(
    weight: torch.Tensor,
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BlockStats]:
    """Reshape a 2D weight tensor into fixed-size blocks for SIGMA."""
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D tensor")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    flat_weight = weight.detach().to(torch.float32).flatten()
    total_elements = int(flat_weight.numel())
    pad = (-total_elements) % block_size
    if pad:
        flat_weight = torch.cat(
            [flat_weight, torch.zeros(pad, dtype=flat_weight.dtype, device=flat_weight.device)]
        )

    blocks = flat_weight.reshape(-1, block_size).contiguous()
    signs = (blocks > 0).to(torch.int8)
    abs_mean = blocks.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    mags = blocks.abs() / abs_mean
    stats = BlockStats(
        n_blocks=int(blocks.shape[0]),
        block_size=int(block_size),
        total_elements=total_elements,
    )
    return signs.cpu(), mags.cpu(), blocks.cpu(), stats


def _pack_bits(bits: torch.Tensor) -> torch.Tensor:
    powers = 1 << torch.arange(bits.shape[1], dtype=torch.long, device=bits.device)
    return (bits.long() * powers).sum(dim=1)


def sketch_random_hadamard(signs: torch.Tensor, n_bits: int, seed: int = 42) -> torch.Tensor:
    """Fixed random orthogonal-style projection from sign bits to bucket ids."""
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(signs.shape[1], n_bits, generator=generator)
    projection = projection / projection.norm(dim=0, keepdim=True).clamp_min(1e-8)
    centered = signs.to(torch.float32) * 2.0 - 1.0
    bits = (centered @ projection > 0).long()
    return _pack_bits(bits).cpu()


def _bucket_codes(n_bits: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(1 << n_bits, dtype=torch.long, device=device)
    shifts = torch.arange(n_bits, dtype=torch.long, device=device)
    return ((values[:, None] >> shifts[None, :]) & 1).to(torch.float32)


def _soft_bucket_probabilities(bit_probs: torch.Tensor) -> torch.Tensor:
    codes = _bucket_codes(bit_probs.shape[1], bit_probs.device)
    probs = bit_probs[:, None, :] * codes[None, :, :] + (1.0 - bit_probs[:, None, :]) * (
        1.0 - codes[None, :, :]
    )
    return probs.clamp_min(1e-8).prod(dim=2)


def _learned_projection(
    signs: torch.Tensor,
    mags: torch.Tensor,
    n_bits: int,
    lr: float = 1e-2,
    steps: int = 250,
    seed: int = 42,
    temperature: float = 0.75,
    progress: bool = True,
) -> torch.Tensor:
    """Learn a differentiable sign projection, then return hard bucket ids."""
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if signs.shape != mags.shape:
        raise ValueError("signs and mags must have matching shapes")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(signs.shape[1], n_bits, generator=generator, requires_grad=True)
    opt = torch.optim.Adam([projection], lr=lr)
    centered = signs.to(torch.float32) * 2.0 - 1.0
    target = mags.to(torch.float32)
    n_buckets = 1 << n_bits

    for step in range(steps):
        logits = centered @ projection
        bit_probs = torch.sigmoid(logits / temperature)
        bucket_probs = _soft_bucket_probabilities(bit_probs)
        bucket_counts = bucket_probs.sum(dim=0).clamp_min(1e-6)
        bucket_means = bucket_probs.T @ target / bucket_counts[:, None]
        expected_means = bucket_probs @ bucket_means
        within_var = (target - expected_means).pow(2).mean()
        balance = (bucket_counts / target.shape[0] - (1.0 / n_buckets)).pow(2).mean()
        entropy = -(bucket_probs * bucket_probs.clamp_min(1e-8).log()).sum(dim=1).mean()
        loss = within_var + 0.05 * balance - 0.001 * entropy

        opt.zero_grad()
        loss.backward()
        opt.step()

        if progress and (step == 0 or (step + 1) % 50 == 0):
            print(f"  [learned] step {step + 1:3d}: within_var={within_var.item():.6f}")

    with torch.no_grad():
        hard_bits = (centered @ projection > 0).long()
    return _pack_bits(hard_bits).cpu()


def sketch_learned_binary(
    signs: torch.Tensor,
    mags: torch.Tensor,
    n_bits: int = 8,
    steps: int = 250,
    seed: int = 42,
    progress: bool = True,
) -> torch.Tensor:
    return _learned_projection(signs, mags, n_bits=n_bits, steps=steps, seed=seed, progress=progress)


def sketch_learned_continuous(
    signs: torch.Tensor,
    mags: torch.Tensor,
    n_bits: int = 8,
    steps: int = 250,
    seed: int = 42,
    progress: bool = True,
) -> torch.Tensor:
    return _learned_projection(signs, mags, n_bits=n_bits, steps=steps, seed=seed, progress=progress)


def within_bucket_variance_ratio(mags: torch.Tensor, buckets: torch.Tensor, n_buckets: int) -> float:
    if n_buckets <= 0:
        raise ValueError("n_buckets must be positive")
    mags = mags.to(torch.float32)
    buckets = buckets.long()
    means = torch.zeros(n_buckets, mags.shape[1], dtype=torch.float32)
    counts = torch.zeros(n_buckets, dtype=torch.float32)
    means.index_add_(0, buckets, mags)
    counts.index_add_(0, buckets, torch.ones_like(buckets, dtype=torch.float32))
    means = means / counts.clamp_min(1.0).unsqueeze(1)
    diffs = mags - means[buckets]
    within_var = diffs.pow(2).mean().item()
    total_var = mags.var(unbiased=False).item()
    return float(within_var / max(total_var, 1e-12))


class SIGMAGenerator(nn.Module):
    """Low-rank magnitude generator conditioned on sketch bucket and tau."""

    def __init__(self, n_buckets: int, n_taus: int, rank: int, block_size: int, seed: int = 0):
        super().__init__()
        self.n_buckets = int(n_buckets)
        self.n_taus = int(n_taus)
        self.rank = int(rank)
        self.block_size = int(block_size)
        self.bucket_embedding = nn.Embedding(n_buckets, rank)
        self.tau_embedding = nn.Embedding(n_taus, rank)
        self.bucket_proj = nn.Linear(rank, block_size, bias=False)
        self.tau_proj = nn.Linear(rank, block_size, bias=False)
        self.interaction_proj = nn.Linear(rank, block_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(block_size))

        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for param in self.parameters():
                if param.ndim > 1:
                    param.normal_(0.0, 0.1, generator=generator)

    def forward(self, bucket_ids: torch.Tensor, tau_ids: torch.Tensor) -> torch.Tensor:
        bucket = self.bucket_embedding(bucket_ids)
        tau = self.tau_embedding(tau_ids)
        magnitude = (
            self.bias
            + self.bucket_proj(bucket)
            + self.tau_proj(tau)
            + self.interaction_proj(bucket * tau)
        )
        return F.softplus(magnitude)


def train_generator(
    signs: torch.Tensor,
    blocks: torch.Tensor,
    buckets: torch.Tensor,
    config: SIGMAConfig,
    lr: float = 1e-3,
    steps: int = 400,
    batch_size: int = 4096,
    device: str | torch.device = "cuda",
    seed: int = 0,
    progress: bool = True,
) -> SIGMAGenerator:
    n_buckets = 1 << config.n_bits
    model = SIGMAGenerator(
        n_buckets=n_buckets,
        n_taus=config.n_taus,
        rank=config.rank,
        block_size=config.block_size,
        seed=seed,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    signs_d = signs.to(device=device, dtype=torch.float32) * 2.0 - 1.0
    blocks_d = blocks.to(device=device, dtype=torch.float32)
    buckets_d = buckets.to(device=device, dtype=torch.long)
    n_blocks = int(blocks_d.shape[0])
    effective_batch = min(int(batch_size), n_blocks)

    for step in range(steps):
        idx = torch.randint(0, n_blocks, (effective_batch,), generator=generator).to(device)
        sign_batch = signs_d[idx]
        weight_batch = blocks_d[idx]
        bucket_batch = buckets_d[idx]
        tau_batch = torch.randint(0, config.n_taus, (effective_batch,), generator=generator).to(device)

        magnitude = model(bucket_batch, tau_batch)
        unscaled = sign_batch * magnitude
        alpha = (weight_batch * unscaled).sum(dim=1) / unscaled.pow(2).sum(dim=1).clamp_min(1e-8)
        recon = alpha.unsqueeze(1) * unscaled
        loss = (weight_batch - recon).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if progress and (step == 0 or (step + 1) % 50 == 0):
            print(f"  [gen] step {step + 1:3d}: mse={loss.item():.6e}")

    return model


def quantize_layer(
    signs: torch.Tensor,
    blocks: torch.Tensor,
    buckets: torch.Tensor,
    model: SIGMAGenerator,
    n_taus: int,
    device: str | torch.device = "cuda",
    chunk: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if chunk <= 0:
        raise ValueError("chunk must be positive")

    model = model.to(device)
    model.eval()
    recon = torch.empty_like(blocks, dtype=torch.float32)
    tau_used = torch.empty(blocks.shape[0], dtype=torch.long)
    alpha_used = torch.empty(blocks.shape[0], dtype=torch.float32)

    with torch.no_grad():
        for start in range(0, blocks.shape[0], chunk):
            end = min(start + chunk, blocks.shape[0])
            sign_batch = signs[start:end].to(device=device, dtype=torch.float32) * 2.0 - 1.0
            weight_batch = blocks[start:end].to(device=device, dtype=torch.float32)
            bucket_batch = buckets[start:end].to(device=device, dtype=torch.long)
            best_loss = torch.full((end - start,), float("inf"), device=device)
            best_tau = torch.zeros(end - start, dtype=torch.long, device=device)
            best_recon = torch.empty_like(weight_batch)
            best_alpha = torch.zeros(end - start, dtype=torch.float32, device=device)

            for tau in range(n_taus):
                tau_batch = torch.full((end - start,), tau, dtype=torch.long, device=device)
                magnitude = model(bucket_batch, tau_batch)
                unscaled = sign_batch * magnitude
                alpha = (weight_batch * unscaled).sum(dim=1) / unscaled.pow(2).sum(dim=1).clamp_min(1e-8)
                candidate = alpha.unsqueeze(1) * unscaled
                loss = (weight_batch - candidate).pow(2).mean(dim=1)
                improved = loss < best_loss
                best_loss = torch.where(improved, loss, best_loss)
                best_tau = torch.where(improved, tau_batch, best_tau)
                best_alpha = torch.where(improved, alpha, best_alpha)
                best_recon[improved] = candidate[improved]

            recon[start:end] = best_recon.cpu()
            tau_used[start:end] = best_tau.cpu()
            alpha_used[start:end] = best_alpha.cpu()

    return recon, tau_used, alpha_used


def activation_weighted_mse(
    weight: torch.Tensor,
    restored: torch.Tensor,
    activation_input: torch.Tensor,
    block_size: int = 64,
    device: str | torch.device = "cuda",
) -> float:
    if weight.shape != restored.shape:
        raise ValueError("weight and restored must have matching shapes")
    if weight.ndim != 2:
        raise ValueError("weight must be 2D")

    _, in_features = weight.shape
    x = activation_input.reshape(-1, activation_input.shape[-1]).to(device=device, dtype=torch.float32)
    x_norms = x.pow(2).sum(dim=0).cpu().clamp_min(1e-8)
    error = (weight.to(torch.float32) - restored.to(torch.float32)).pow(2)
    blocks_per_row = math.ceil(in_features / block_size)
    padded_cols = blocks_per_row * block_size
    if padded_cols != in_features:
        padded_error = torch.zeros((weight.shape[0], padded_cols), dtype=error.dtype)
        padded_error[:, :in_features] = error
        error = padded_error
        x_pad = torch.zeros(padded_cols, dtype=x_norms.dtype)
        x_pad[:in_features] = x_norms
        x_norms = x_pad

    block_mse = error.reshape(weight.shape[0], blocks_per_row, block_size).mean(dim=2)
    input_weights = x_norms.reshape(blocks_per_row, block_size).mean(dim=1)
    input_weights = input_weights / input_weights.mean().clamp_min(1e-8)
    return float((block_mse * input_weights.unsqueeze(0)).mean().item())


def sigma_bpw(
    n_blocks: int,
    n_buckets: int,
    n_taus: int,
    rank: int,
    block_size: int,
    scale_bits: int = 6,
    tau_bits: int = 3,
    rescue_bits: int = 1,
    param_bits: int = 16,
) -> float:
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive")
    per_block_bits = block_size + scale_bits + tau_bits + rescue_bits
    generator_params = (n_buckets + n_taus) * rank + 3 * rank * block_size + block_size
    total_bits = n_blocks * per_block_bits + generator_params * param_bits
    return float(total_bits / (n_blocks * block_size))
