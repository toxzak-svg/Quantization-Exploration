"""Benchmark RFF HSIC vs RBF HSIC on realistic-shaped data."""
import sys
import time
sys.path.insert(0, '.')

import torch
from src.cross_layer_mi import hsic_matrix


def make_synthetic_acts(n_layers: int, n_tokens: int, hidden: int, seed: int = 0):
    """Build a CalibrationActivations-shaped dict."""
    gen = torch.Generator().manual_seed(seed)
    acts = {}
    centre = (n_layers - 1) / 2.0
    sigma = n_layers / 4.0
    x0 = torch.randn(1, n_tokens, hidden, generator=gen)
    x = x0.clone()
    for i in range(n_layers):
        bell = float(torch.tensor(0.95).exp().item()
                     ** ((i - centre) ** 2 / (2.0 * sigma ** 2)))
        noise = 0.4 * torch.randn(1, n_tokens, hidden, generator=gen)
        x = bell * x + (1.0 - bell) * x0 + noise
        acts[i] = x.detach().to(torch.float32)
    from src.cross_layer_mi import CalibrationActivations
    return CalibrationActivations(hidden_states=acts, layer_keys=[f"l{i}" for i in range(n_layers)], n_tokens=n_tokens)


def bench(name, fn, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.time()
        out = fn()
        times.append(time.time() - t0)
    best = min(times)
    median = sorted(times)[len(times) // 2]
    print(f"  {name:40s}  best={best*1000:7.1f}ms  median={median*1000:7.1f}ms")
    return best, out


# Realistic-shape workload
print("=== Gemma 4 E2B text-decoder shape: 35 layers, 512 tokens, hidden=1536 ===")
acts = make_synthetic_acts(35, 512, 1536)
print()

t_rbf, m_rbf = bench("RBF (exact, full L^2)", lambda: hsic_matrix(acts, sub_sample=512, method="rbf"))
t_rbf_h, m_rbf_h = bench("RBF (horizon=4)", lambda: hsic_matrix(acts, sub_sample=512, method="rbf", horizon=4))
t_rff64_1, m_rff64_1 = bench("RFF 64   samples=1 full", lambda: hsic_matrix(acts, sub_sample=512, method="rff", n_rff=64, n_rff_samples=1))
t_rff64_1_h, m_rff64_1_h = bench("RFF 64   samples=1 horizon=4", lambda: hsic_matrix(acts, sub_sample=512, method="rff", n_rff=64, n_rff_samples=1, horizon=4))
t_rff128_1, m_rff128_1 = bench("RFF 128  samples=1 full", lambda: hsic_matrix(acts, sub_sample=512, method="rff", n_rff=128, n_rff_samples=1))
t_rff128_1_h, m_rff128_1_h = bench("RFF 128  samples=1 horizon=4", lambda: hsic_matrix(acts, sub_sample=512, method="rff", n_rff=128, n_rff_samples=1, horizon=4))
t_rff256_1, m_rff256_1 = bench("RFF 256  samples=1 full", lambda: hsic_matrix(acts, sub_sample=512, method="rff", n_rff=256, n_rff_samples=1))
t_rff256_1_h, m_rff256_1_h = bench("RFF 256  samples=1 horizon=4", lambda: hsic_matrix(acts, sub_sample=512, method="rff", n_rff=256, n_rff_samples=1, horizon=4))
t_rff256_4, m_rff256_4 = bench("RFF 256  samples=4 full", lambda: hsic_matrix(acts, sub_sample=512, method="rff", n_rff=256, n_rff_samples=4))
t_rff256_4_h, m_rff256_4_h = bench("RFF 256  samples=4 horizon=4", lambda: hsic_matrix(acts, sub_sample=512, method="rff", n_rff=256, n_rff_samples=4, horizon=4))
t_rff1024_1, m_rff1024_1 = bench("RFF 1024 samples=1 full", lambda: hsic_matrix(acts, sub_sample=512, method="rff", n_rff=1024, n_rff_samples=1))

print()
print(f"Speedup vs RBF: RFF 64  = {t_rbf / t_rff64_1:.1f}x")
print(f"Speedup vs RBF: RFF 128 = {t_rbf / t_rff128_1:.1f}x")
print(f"Speedup vs RBF: RFF 256 = {t_rbf / t_rff256_1:.1f}x")
print(f"Speedup vs RBF: RFF 256x4 = {t_rbf / t_rff256_4:.1f}x")
print(f"Speedup vs RBF: RFF 1024 = {t_rbf / t_rff1024_1:.1f}x")

# Horizon speedup
print()
print("=== Horizon speedup (24% of entries computed) ===")
print(f"RBF full -> RBF horizon:    {t_rbf / t_rbf_h:.2f}x")
print(f"RFF 64  full -> horizon:    {t_rff64_1 / t_rff64_1_h:.2f}x")
print(f"RFF 128 full -> horizon:    {t_rff128_1 / t_rff128_1_h:.2f}x")
print(f"RFF 256 full -> horizon:    {t_rff256_1 / t_rff256_1_h:.2f}x")
print(f"RFF 256x4 full -> horizon:  {t_rff256_4 / t_rff256_4_h:.2f}x")

print()
print(f"Best overall vs RBF full: RFF 256 samples=1 horizon=4: {t_rbf / t_rff256_1_h:.2f}x")

print()
print("=== Agreement with RBF (matrix-level correlation) ===")
def corr(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item())
print(f"RFF 64  x1: corr={corr(m_rbf, m_rff64_1):.4f}")
print(f"RFF 128 x1: corr={corr(m_rbf, m_rff128_1):.4f}")
print(f"RFF 256 x1: corr={corr(m_rbf, m_rff256_1):.4f}")
print(f"RFF 256 x4: corr={corr(m_rbf, m_rff256_4):.4f}")
print(f"RFF 1024x1: corr={corr(m_rbf, m_rff1024_1):.4f}")

print()
print("=== Kendall tau of per-layer MI ranking vs RBF ===")
from src.cross_layer_mi import mi_scores_from_matrix
scores_rbf = mi_scores_from_matrix(m_rbf, horizon=4)
order_rbf = torch.argsort(scores_rbf, descending=True)
rank_rbf = {idx: r for r, idx in enumerate(order_rbf.tolist())}
for label, m in [("RFF 64", m_rff64_1), ("RFF 128", m_rff128_1), ("RFF 256x1", m_rff256_1), ("RFF 256x4", m_rff256_4), ("RFF 1024", m_rff1024_1)]:
    s = mi_scores_from_matrix(m, horizon=4)
    order = torch.argsort(s, descending=True).tolist()
    n = len(order)
    conc = disc = 0
    for i in range(n):
        for j in range(i+1, n):
            a = rank_rbf[order[i]] - rank_rbf[order[j]]
            b = order.index(order[i]) - order.index(order[j])
            if (a > 0) == (b > 0):
                conc += 1
            else:
                disc += 1
    pairs = max(conc + disc, 1)
    tau = (conc - disc) / pairs
    print(f"  {label:15s} tau = {tau:+.4f}")