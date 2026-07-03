"""End-to-end: cross-layer MI on Gemma 4 E2B calibration.

Loads the model (text-only path; vision/audio towers are unused but loaded
so the safetensors parse), runs a single calibration forward pass on
WikiText, captures per-layer activations, computes the HSIC matrix, and
dumps:

* The full L x L HSIC matrix as JSON and as a markdown table snippet.
* Per-layer MI scores (horizon-windowed) ranked top to bottom.
* Bit allocation derived from those scores at the configured target BPW.
* A side-by-side comparison against the per-layer sigma score.

Designed to run on CPU when CUDA is unavailable; for the 35-layer E2B
text decoder one forward pass over ~128 tokens takes a few minutes on
a workstation CPU. Reduce ``calib_tokens`` to smoke-test faster.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Allow running as ``python scripts/run_cross_layer_mi.py`` from the repo
# root without setting PYTHONPATH explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma4ForCausalLM

from src.cross_layer_mi import (
    CalibrationActivations,
    allocate_bits,
    sigma_scores_from_activations,
)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def save_activation_cache(
    path: Path,
    *,
    hidden_states: dict[int, torch.Tensor],
    layer_keys: list[str],
    n_tokens: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "hidden_states": hidden_states,
            "layer_keys": layer_keys,
            "n_tokens": n_tokens,
        },
        tmp_path,
    )
    tmp_path.replace(path)


def write_progress(
    path: Path | None,
    *,
    stage: str,
    complete: bool,
    details: dict[str, Any] | None = None,
) -> None:
    if path is None:
        return
    write_json_atomic(
        path,
        {
            "stage": stage,
            "complete": complete,
            "timestamp_utc": int(time.time()),
            "details": details or {},
        },
    )


def _load_text_decoder(model_path: str, dtype: torch.dtype) -> torch.nn.Module:
    """Load Gemma 4 E2B's text-only decoder.

    Uses ``Gemma4ForCausalLM`` directly, which skips the vision and audio
    towers that the multimodal ``Gemma4ForConditionalGeneration`` would
    pull in. For a 2B-param model the safetensors file is ~10 GB on disk;
    the text-only class needs roughly half that in RAM.
    """
    full = Gemma4ForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    full.eval()
    return full


def _capture_hidden_states(text_decoder: torch.nn.Module) -> list:
    """Register hooks on every transformer block to capture its output.

    Returns a list of registered hooks so the caller can remove them.
    """
    captured: dict[int, list[torch.Tensor]] = {}

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            # Decoder layers typically return a tuple where index 0 is the
            # hidden states tensor. Some return a ModelOutput; we handle
            # both.
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = getattr(output, "last_hidden_state", output)
            if isinstance(tensor, torch.Tensor):
                captured[idx] = tensor.detach().to("cpu", torch.float32)
        return hook

    handles = []
    # Discover the layer list. Different transformers versions name it
    # differently; check the most common options.
    layers = None
    for attr in ("layers", "h", "transformer", "model"):
        candidate = getattr(text_decoder, attr, None)
        if candidate is not None and hasattr(candidate, "__len__"):
            try:
                if len(candidate) > 0:
                    layers = candidate
                    break
            except TypeError:
                continue
    if layers is None:
        # Fallback: walk named_modules and find the deepest ModuleList.
        for _name, module in text_decoder.named_modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) > 4:
                layers = module
                break
    if layers is None:
        raise RuntimeError("could not locate transformer layer list")
    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))
    return handles, captured, len(layers)


def collect_calibration_activations(
    model_path: str,
    calib_text: str,
    calib_tokens: int = 256,
    cache_path: Path | None = None,
    synth_n_layers: int | None = None,
    synth_hidden: int | None = None,
) -> CalibrationActivations:
    """Run one calibration forward pass and return per-layer activations.

    Parameters
    ----------
    model_path : str
        Path or HF id for the Gemma 4 model directory.
    calib_text : str
        Raw calibration text. Will be tokenised and truncated.
    calib_tokens : int
        Maximum sequence length.
    cache_path : Path, optional
        If provided, the captured hidden_states dict is cached here on
        completion so subsequent runs can skip the forward pass.
    synth_n_layers, synth_hidden : int, optional
        If set, skip the model entirely and generate synthetic
        activations of the requested shape. Used for end-to-end
        pipeline testing on machines that can't host the full model.

    Returns
    -------
    CalibrationActivations
        Map of layer index to (batch, seq, hidden) tensor plus metadata.
    """
    if cache_path and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return CalibrationActivations(
            hidden_states=payload["hidden_states"],
            layer_keys=payload["layer_keys"],
            n_tokens=payload["n_tokens"],
        )

    if synth_n_layers and synth_hidden:
        return _synth_activations(
            n_tokens=calib_tokens,
            n_layers=synth_n_layers,
            hidden=synth_hidden,
            cache_path=cache_path,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    full_model = _load_text_decoder(model_path, torch.bfloat16)
    text_decoder = getattr(full_model, "model", full_model)
    handles, captured, n_layers = _capture_hidden_states(text_decoder)
    print(f"[calib] captured hooks on {n_layers} layers")

    encodings = tokenizer(
        calib_text,
        return_tensors="pt",
        truncation=True,
        max_length=calib_tokens,
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    n_tokens = int(input_ids.numel())
    print(f"[calib] forward pass: {n_tokens} tokens")

    t0 = time.time()
    with torch.no_grad():
        _ = full_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    dt = time.time() - t0
    print(f"[calib] forward done in {dt:.1f}s; captured {len(captured)} layer tensors")

    for handle in handles:
        handle.remove()

    if not captured:
        raise RuntimeError("no hidden states captured -- check decoder structure")

    layer_keys = [f"layer_{i}" for i in range(n_layers)]
    n_tokens_total = int(next(iter(captured.values())).shape[1])

    acts = CalibrationActivations(
        hidden_states=captured,
        layer_keys=layer_keys,
        n_tokens=n_tokens_total,
    )

    if cache_path:
        save_activation_cache(
            cache_path,
            hidden_states=captured,
            layer_keys=layer_keys,
            n_tokens=n_tokens_total,
        )
        print(f"[calib] cached activations to {cache_path}")

    return acts


def _synth_activations(
    n_tokens: int,
    n_layers: int,
    hidden: int,
    cache_path: Path | None,
    seed: int = 0,
) -> CalibrationActivations:
    """Generate realistic-looking synthetic layer activations.

    Each layer i is built as:

        x_0 = N(0, I)
        x_{i+1} = bell_i * x_i + (1 - bell_i) * x_0 + alpha_i * noise_i

    where ``bell_i`` is a bell-shaped coupling centred in the middle of
    the network (so centre layers are tightly coupled to their
    neighbours) and ``alpha_i`` grows linearly toward the ends (so the
    edge layers are noisier). A few "hub" layers in the middle are
    boosted to mimic the empirical observation that a small subset of
    transformer layers carry information read by many downstream
    consumers.

    Sigma scores trend opposite (largest at the noisy edges), so
    Kendall tau between MI and sigma ranking is meaningfully negative
    on this synthetic -- the pattern we expect to see on real Gemma.
    """
    gen = torch.Generator().manual_seed(seed)
    activations: dict[int, torch.Tensor] = {}
    x0 = torch.randn(1, n_tokens, hidden, generator=gen)
    x = x0.clone()
    centre = (n_layers - 1) / 2.0
    sigma_layers = n_layers / 5.0
    # Mark three hub layers with extra incoming coupling.
    hub_layers = {n_layers // 3, n_layers // 2, 2 * n_layers // 3}
    for i in range(n_layers):
        bell_base = float(
            torch.tensor(0.95).exp().item()
            ** ((i - centre) ** 2 / (2.0 * sigma_layers ** 2))
        )
        bell = min(0.99, bell_base + (0.04 if i in hub_layers else 0.0))
        alpha = 0.05 + 0.7 * (abs(i - centre) / max(centre, 1.0)) ** 1.5
        noise = alpha * torch.randn(1, n_tokens, hidden, generator=gen)
        x = bell * x + (1.0 - bell) * x0 + noise
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        x = x / rms
        activations[i] = x.detach().to(torch.float32)
    layer_keys = [f"layer_{i}" for i in range(n_layers)]
    acts = CalibrationActivations(
        hidden_states=activations,
        layer_keys=layer_keys,
        n_tokens=n_tokens,
    )
    if cache_path:
        save_activation_cache(
            cache_path,
            hidden_states=activations,
            layer_keys=layer_keys,
            n_tokens=n_tokens,
        )
        print(f"[synth] cached activations to {cache_path}")
    return acts


def _fmt_row(name: str, score: float, rank: int) -> str:
    return f"| {rank:2d} | {name:>8s} | {score:8.4f} |"


def report_layer_indices(acts: CalibrationActivations, conditioning: str | None) -> list[int]:
    indices = sorted(int(idx) for idx in acts.hidden_states)
    if conditioning == "delta":
        return indices[1:]
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/gemma-4-E2B")
    parser.add_argument(
        "--calib-data",
        default="data/wiki.test.txt",
        help="path to calibration text file",
    )
    parser.add_argument("--calib-tokens", type=int, default=256)
    parser.add_argument("--cache", type=Path, default=Path("eval_results/calib_activations.pt"))
    parser.add_argument("--output", type=Path, default=Path("eval_results/cross_layer_mi_report.json"))
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--bits-min", type=float, default=1.5)
    parser.add_argument("--bits-max", type=float, default=8.0)
    parser.add_argument("--target-bpw", type=float, default=4.0)
    parser.add_argument("--sub-sample", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--conditioning",
        choices=[None, "delta"],
        default=None,
        help="Phase 3 conditional MI. 'delta' preprocesses activations through "
        "residual_deltas so the matrix approximates "
        "I(subblock_l; subblock_{l+k} | x_{l-1}).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the HSIC matrix build ('cpu' or 'cuda'). "
        "Defaults to cpu; pass 'cuda' on Colab for the GPU path.",
    )
    parser.add_argument(
        "--synth",
        action="store_true",
        help="skip model load; generate synthetic activations of the requested shape",
    )
    parser.add_argument("--synth-layers", type=int, default=35)
    parser.add_argument("--synth-hidden", type=int, default=1536)
    args = parser.parse_args()
    progress_output = args.progress_output or args.output.with_suffix(".progress.json")

    if args.synth:
        print("[main] synthetic-activation mode: skipping model load")
        write_progress(
            progress_output,
            stage="collecting_synthetic_activations",
            complete=False,
            details={"cache": str(args.cache), "calib_tokens": args.calib_tokens},
        )
        acts = _synth_activations(
            n_tokens=args.calib_tokens,
            n_layers=args.synth_layers,
            hidden=args.synth_hidden,
            cache_path=args.cache,
            seed=args.seed,
        )
    else:
        calib_path = Path(args.calib_data)
        calib_text = calib_path.read_text(encoding="utf-8", errors="replace")[:200_000]
        print(f"[main] loading {len(calib_text)} chars from {calib_path}")
        write_progress(
            progress_output,
            stage="collecting_model_activations",
            complete=False,
            details={
                "cache": str(args.cache),
                "calib_data": str(calib_path),
                "calib_tokens": args.calib_tokens,
            },
        )
        acts = collect_calibration_activations(
            args.model_path,
            calib_text,
            calib_tokens=args.calib_tokens,
            cache_path=args.cache,
        )

    print(
        f"[main] activations: {acts.layer_count()} layers, "
        f"{acts.n_tokens} tokens, hidden={next(iter(acts.hidden_states.values())).shape[-1]}"
    )
    write_progress(
        progress_output,
        stage="allocating_bits",
        complete=False,
        details={
            "layers": acts.layer_count(),
            "tokens": acts.n_tokens,
            "device": args.device,
            "conditioning": args.conditioning,
        },
    )

    alloc = allocate_bits(
        acts,
        horizon=args.horizon,
        bits_min=args.bits_min,
        bits_max=args.bits_max,
        target_avg_bpw=args.target_bpw,
        sub_sample=args.sub_sample,
        seed=args.seed,
        conditioning=args.conditioning,
        device=args.device,
    )

    sigma_scores = sigma_scores_from_activations(acts)

    # Rank by MI descending; track sigma rank for the comparison.
    mi_rank_order = torch.argsort(alloc.mi_scores, descending=True).tolist()
    sigma_rank_order = torch.argsort(sigma_scores, descending=True).tolist()
    mi_rank = {idx: rank for rank, idx in enumerate(mi_rank_order)}
    sigma_rank = {idx: rank for rank, idx in enumerate(sigma_rank_order)}

    print("\n=== Per-layer MI allocation ===")
    print("| rank | layer    | mi_score | bits | sigma_score | sigma_rank |")
    print("|------|----------|----------|------|-------------|------------|")
    for rank, idx in enumerate(mi_rank_order):
        print(
            f"| {rank:4d} | {idx:>8d} | {float(alloc.mi_scores[idx]):8.4f} "
            f"| {float(alloc.bits[idx]):5.2f} | {float(sigma_scores[idx]):8.4f} "
            f"| {sigma_rank[idx]:10d} |"
        )

    # Kendall's tau between MI and sigma rankings: tells us how different
    # the cross-layer signal is from the per-layer static one.
    n = len(mi_rank_order)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            mi_ord = (mi_rank[mi_rank_order[i]] - mi_rank[mi_rank_order[j]]) > 0
            sig_ord = (sigma_rank[mi_rank_order[i]] - sigma_rank[mi_rank_order[j]]) > 0
            if mi_ord == sig_ord:
                concordant += 1
            else:
                discordant += 1
    pairs = max(concordant + discordant, 1)
    kendall_tau = (concordant - discordant) / pairs
    print(f"\nKendall tau(MI rank, sigma rank) = {kendall_tau:.4f}")

    # Top-10 / bottom-10 by MI: where would MI vs sigma disagree most?
    top10 = mi_rank_order[:10]
    bot10 = mi_rank_order[-10:]
    print("\nTop-10 MI layers (rank | layer | sigma_rank):")
    for rank, idx in enumerate(top10):
        print(f"  {rank:2d}  layer {idx:>3d}  sigma_rank={sigma_rank[idx]:3d}")
    print("\nBottom-10 MI layers (rank | layer | sigma_rank):")
    for rank_in_bot, idx in enumerate(bot10):
        print(f"  {rank_in_bot:2d}  layer {idx:>3d}  sigma_rank={sigma_rank[idx]:3d}")

    summary = alloc.summary()
    summary["kendall_tau_vs_sigma"] = float(kendall_tau)
    summary["bits_per_layer"] = [float(b) for b in alloc.bits.tolist()]
    summary["mi_scores"] = [float(s) for s in alloc.mi_scores.tolist()]
    summary["layer_indices"] = report_layer_indices(acts, args.conditioning)
    summary["sigma_scores"] = [float(s) for s in sigma_scores.tolist()]
    summary["calib_tokens"] = acts.n_tokens
    summary["horizon"] = args.horizon
    summary["target_avg_bpw"] = args.target_bpw

    write_json_atomic(args.output, summary)
    write_progress(
        progress_output,
        stage="complete",
        complete=True,
        details={"output": str(args.output), "avg_bits": summary["avg_bits"]},
    )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
