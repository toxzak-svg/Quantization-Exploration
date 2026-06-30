"""Collect activation-derived layer weights for mixed-budget allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch


def module_weight_key(module_name: str) -> str:
    return f"{module_name}.weight"


def atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


class ActivationAccumulator:
    def __init__(self) -> None:
        self._sum_square: dict[str, float] = {}
        self._samples: dict[str, int] = {}

    def record(self, key: str, activation: torch.Tensor) -> None:
        values = activation.detach().to(dtype=torch.float32)
        self._sum_square[key] = self._sum_square.get(key, 0.0) + values.square().sum().item()
        self._samples[key] = self._samples.get(key, 0) + values.numel()

    def to_stats(self) -> dict[str, dict]:
        stats = {}
        for key in sorted(self._sum_square):
            samples = self._samples[key]
            stats[key] = {
                "samples": samples,
                "sum_square": self._sum_square[key],
                "mean_square": self._sum_square[key] / samples if samples else 0.0,
            }
        return stats

    def save(self, stats_path: Path, weights_path: Path) -> None:
        stats = self.to_stats()
        atomic_write_json(stats_path, stats)
        atomic_write_json(weights_path, normalize_activation_weights(stats))


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


def normalize_activation_weights(stats: dict[str, dict]) -> dict[str, float]:
    """Normalize per-layer mean-square activations to a robust center of 1.0.

    Raw mean-square activations span 10+ orders of magnitude across a multimodal
    model: embed_tokens dominate the input, RMSNorm layers collapse downstream
    signal, vision/audio towers have wildly different statistics. Naive
    mean-of-all division makes 99% of language-model layers look irrelevant
    (activation weight 1e-9), which causes the mixed-budget allocator to
    budget-cheap formats on the wrong layers and crater PPL.

    The fix: compute the center using only the 2D weight-module activations
    (no embeddings, no norms, no towers), use a winsorized mean (trim top and
    bottom 10%) to suppress outliers within that set, and clamp the final
    weights to [0.05, 20.0] so no single layer can dominate or vanish.
    """
    raw = {
        key: float(item.get("mean_square", 0.0))
        for key, item in stats.items()
        if float(item.get("mean_square", 0.0)) > 0.0
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


def iter_weighted_linear_modules(model, include_substring: str) -> Iterable[tuple[str, torch.nn.Module]]:
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        key = module_weight_key(name)
        if include_substring and include_substring not in key:
            continue
        yield key, module


def attach_hooks(model, accumulator: ActivationAccumulator, include_substring: str):
    handles = []
    for key, module in iter_weighted_linear_modules(model, include_substring):
        def hook(_module, inputs, weight_key=key):
            if inputs:
                accumulator.record(weight_key, inputs[0])

        handles.append(module.register_forward_pre_hook(hook))
    return handles


def collect_activation_weights(args: argparse.Namespace) -> None:
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
    model.eval()

    text = Path(args.wikitext).read_text(encoding="utf-8")
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    if args.tokens:
        input_ids = input_ids[:, : args.tokens]

    accumulator = ActivationAccumulator()
    handles = attach_hooks(model, accumulator, include_substring=args.include_substring)
    if not handles:
        raise RuntimeError(f"No 2D weight modules matched {args.include_substring!r}")

    chunks = 0
    try:
        with torch.no_grad():
            for start in range(0, max(input_ids.shape[1] - 1, 1), args.stride):
                end = min(start + args.max_length, input_ids.shape[1])
                if end - start < 2:
                    break
                model(input_ids[:, start:end].to(device))
                chunks += 1
                if chunks % args.save_every_chunks == 0:
                    accumulator.save(Path(args.stats_output), Path(args.output))
                    atomic_write_json(
                        Path(args.progress_output),
                        {
                            "chunks": chunks,
                            "tokens_seen": int(end),
                            "output": args.output,
                            "stats_output": args.stats_output,
                        },
                    )
    finally:
        for handle in handles:
            handle.remove()

    accumulator.save(Path(args.stats_output), Path(args.output))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/gemma-4-E2B")
    parser.add_argument("--wikitext", default="data/wiki.test.txt")
    parser.add_argument("--output", default="eval_results/activation_weights_gemma4.json")
    parser.add_argument("--stats-output", default="eval_results/activation_stats_gemma4.json")
    parser.add_argument("--progress-output", default="eval_results/activation_weights_progress.json")
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--save-every-chunks", type=int, default=1)
    parser.add_argument("--include-substring", default="language_model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    collect_activation_weights(parse_args())


if __name__ == "__main__":
    main()
