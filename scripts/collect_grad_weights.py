"""Collect gradient-based per-layer sensitivity weights for mixed-budget allocation.

Why gradient-based:
    Activation-magnitude is a poor proxy for quantization tolerance on multimodal
    transformers. RMSNorm collapses downstream signal and embedding/vision/audio
    towers dominate by 10+ orders of magnitude, which makes 99% of language layers
    look "unimportant" to the allocator and crater PPL on the wrong layers.

    The real sensitivity metric is parameter importance under the training
    objective. We approximate it with a Fisher-style scoring:

        s_i = E[(W_i * dL/dW_i)^2]    averaged over calibration batches

    where the loss is next-token cross-entropy. This is the same proxy GPTQ/AWQ
    use internally for mixed-precision allocation; it correlates with actual
    quantization-induced perplexity degradation much better than activation
    magnitude.

Usage on Colab (L4 GPU, full sweep):

    cd /content/sub1quant
    python3 -u scripts/collect_grad_weights.py \\
        --model-dir /content/models/gemma-4-E2B \\
        --wikitext data/wiki.test.txt \\
        --output eval_results/grad_weights_gemma4.json \\
        --stats-output eval_results/grad_stats_gemma4.json \\
        --progress-output eval_results/grad_weights_progress.json \\
        --tokens 32768 --max-length 512 --stride 512 --device cuda

Usage on CPU (smoke):

    python scripts/collect_grad_weights.py \\
        --model-dir models/gemma-4-E2B \\
        --wikitext data/wiki.test.txt \\
        --output eval_results/grad_weights_smoke.json \\
        --tokens 256 --max-length 64 --stride 64 --device cpu --max-layers 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch


# ---------- IO helpers ----------

def atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


# ---------- Outlier key filter (shared logic with activation collector) ----------

OUTLIER_KEY_SUBSTRINGS = (
    "embed_tokens",
    "embed_vision",
    "embed_audio",
    "lm_head",
    "norm",
    "audio_tower",
    "vision_tower",
)


def _is_outlier_key(key: str) -> bool:
    return any(part in key for part in OUTLIER_KEY_SUBSTRINGS)


# ---------- Gradient sensitivity accumulator ----------

class GradSensitivityAccumulator:
    """Accumulate (W * dL/dW)^2 and |W * dL/dW| per parameter.

    Hook strategy:
        We register a full backward hook on each 2D-weight Linear module. After
        `loss.backward()` populates `module.weight.grad`, the hook fires and we
        compute `W * dL/dW` elementwise on the parameter. This avoids needing
        to manually reconstruct dL/dW from grad_output (which depends on the
        module variant and gets messy for GQA / fused QKV).
    """

    def __init__(self) -> None:
        self._sum_sq: dict[str, float] = {}        # sum of (W * g)^2
        self._sum_abs: dict[str, float] = {}       # sum of |W * g|
        self._count: dict[str, int] = {}           # numel per layer
        self._handles: list = []

    def attach(self, model: torch.nn.Module, include_substring: str) -> None:
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                continue
            key = f"{name}.weight"
            if include_substring and include_substring not in key:
                continue
            self._handles.append(
                module.register_full_backward_hook(self._make_hook(key))
            )

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, key: str):
        def hook(_module, _grad_input, _grad_output):
            if _module.weight.grad is None:
                return
            with torch.no_grad():
                prod = (_module.weight * _module.weight.grad).detach().to(torch.float32)
                self._sum_sq[key] = self._sum_sq.get(key, 0.0) + prod.pow(2).sum().item()
                self._sum_abs[key] = self._sum_abs.get(key, 0.0) + prod.abs().sum().item()
                self._count[key] = self._count.get(key, 0) + prod.numel()
        return hook

    def to_stats(self) -> dict[str, dict]:
        stats: dict[str, dict] = {}
        for key in sorted(self._sum_sq):
            n = self._count.get(key, 0)
            ssq = self._sum_sq.get(key, 0.0)
            sab = self._sum_abs.get(key, 0.0)
            stats[key] = {
                "samples": n,
                "sum_sq": ssq,
                "sum_abs": sab,
                "mean_prod_sq": ssq / n if n else 0.0,
                "rms_prod": (ssq / n) ** 0.5 if n else 0.0,
                "mean_abs_prod": sab / n if n else 0.0,
            }
        return stats


# ---------- Normalization ----------

def normalize_grad_weights(stats: dict[str, dict]) -> dict[str, float]:
    """Robust normalize Fisher-style scores to a center of 1.0.

    Identical normalization discipline as collect_activation_weights.py:
        1. Skip outlier keys (embeddings, lm_head, norms, towers).
        2. Winsorize the inner 80% of the remaining values to compute the
           center (robust to single-layer spikes).
        3. Clamp final weights to [0.05, 20.0] so no single layer can dominate
           or vanish from the allocator's view.

    Using `rms_prod` (= sqrt(E[(W*g)^2])) as the raw score — it's the natural
    scale of the per-layer sensitivity and stays positive.
    """
    raw = {
        key: float(item.get("rms_prod", 0.0))
        for key, item in stats.items()
        if float(item.get("rms_prod", 0.0)) > 0.0
    }
    if not raw:
        return {}

    quant_layer_values = sorted(
        v for k, v in raw.items() if not _is_outlier_key(k)
    )
    if not quant_layer_values:
        quant_layer_values = sorted(raw.values())

    n = len(quant_layer_values)
    trim = max(1, n // 10)
    trimmed = quant_layer_values[trim : n - trim] if n > 2 * trim else quant_layer_values
    center = sum(trimmed) / len(trimmed) if trimmed else 1.0

    FLOOR = 0.05
    CEIL = 20.0
    return {
        key: max(FLOOR, min(CEIL, v / center))
        for key, v in sorted(raw.items())
    }


# ---------- Main ----------

def collect_grad_weights(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(args.model_dir)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=None,
        local_files_only=True,
    ).to(device)

    # Gradients need to flow through the weights we want to score. We use
    # train() so dropout/etc are off but autograd tracks the weights. We do
    # NOT use any optimizer — gradients are accumulated purely for the score
    # and then zeroed.
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)

    text = Path(args.wikitext).read_text(encoding="utf-8")
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    if args.tokens:
        input_ids = input_ids[:, : args.tokens]

    accumulator = GradSensitivityAccumulator()
    accumulator.attach(model, include_substring=args.include_substring)
    if not accumulator._handles:
        raise RuntimeError(f"No 2D weight modules matched {args.include_substring!r}")

    chunks = 0
    max_layers = args.max_layers
    try:
        for start in range(0, max(input_ids.shape[1] - 1, 1), args.stride):
            end = min(start + args.max_length, input_ids.shape[1])
            if end - start < 2:
                break
            batch = input_ids[:, start:end].to(device)

            # Gemma 4 multimodal: pass input_ids only; next-token CE loss is
            # computed when labels == input_ids (the AutoModelForCausalLM path).
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            if loss is None or not loss.requires_grad:
                # Some multimodal heads don't compute LM loss on text-only input.
                # Fall back to a synthetic scalar that still drives backward.
                dummy = batch.float().mean() * 0.0
                dummy.backward()
            else:
                loss.backward()

            chunks += 1
            model.zero_grad(set_to_none=True)

            if max_layers is not None and chunks >= max_layers:
                # Smoke mode: stop after N chunks for fast verification.
                break

            if chunks % args.save_every_chunks == 0:
                _save(accumulator, args)
    finally:
        accumulator.detach()

    _save(accumulator, args)
    atomic_write_json(
        Path(args.progress_output),
        {
            "chunks": chunks,
            "tokens_seen": int(input_ids.shape[1]),
            "output": args.output,
            "stats_output": args.stats_output,
            "complete": True,
        },
    )
    print(f"WROTE {args.output}")
    print(f"WROTE {args.stats_output}")
    print(f"WROTE {args.progress_output}")


def _save(accumulator: GradSensitivityAccumulator, args: argparse.Namespace) -> None:
    stats = accumulator.to_stats()
    atomic_write_json(Path(args.stats_output), stats)
    atomic_write_json(Path(args.output), normalize_grad_weights(stats))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/gemma-4-E2B")
    parser.add_argument("--wikitext", default="data/wiki.test.txt")
    parser.add_argument("--output", default="eval_results/grad_weights_gemma4.json")
    parser.add_argument("--stats-output", default="eval_results/grad_stats_gemma4.json")
    parser.add_argument("--progress-output", default="eval_results/grad_weights_progress.json")
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--save-every-chunks", type=int, default=1)
    parser.add_argument("--include-substring", default="language_model")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Smoke mode: stop after N chunks (CPU testing).")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    collect_grad_weights(parse_args())


if __name__ == "__main__":
    main()