"""Publication-grade WikiText perplexity runner for quantized checkpoints.

This script is intentionally stricter than the short smoke benchmark:

* ``--tokens all`` evaluates the full tokenized WikiText file.
* Base and quantized runs are recorded in one JSON payload.
* Weight-application lists are compacted to exact counts plus small samples.
* Optional expected replacement/skip counts fail the run if the checkpoint is
  not applied through the intended model path.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_quantized import apply_quantized_weights


def parse_token_limit(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"all", "full", "none", "0"}:
        return None

    tokens = int(normalized)
    if tokens <= 0:
        return None
    return tokens


def compact_apply_stats(stats: dict[str, Any], sample_size: int = 5) -> dict[str, Any]:
    skipped = list(stats.get("skipped") or [])
    missing = list(stats.get("missing") or [])
    shape_mismatches = list(stats.get("shape_mismatches") or [])
    return {
        "replaced": int(stats.get("replaced", 0)),
        "skipped_count": len(skipped),
        "missing_count": len(missing),
        "shape_mismatch_count": len(shape_mismatches),
        "skipped_sample": skipped[:sample_size],
        "missing_sample": missing[:sample_size],
        "shape_mismatch_sample": shape_mismatches[:sample_size],
    }


def eval_ppl(
    model,
    tokenizer,
    text: str,
    device: str,
    token_limit: int | None,
    max_length: int,
    stride: int,
) -> dict[str, Any]:
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"]
    if token_limit is not None:
        input_ids = input_ids[:, :token_limit]
    input_ids = input_ids.to(device)

    seq_len = int(input_ids.shape[1])
    if seq_len < 2:
        raise ValueError(f"Need at least 2 tokens for perplexity, got {seq_len}")

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        batch = input_ids[:, begin_loc:end_loc]
        target = batch.clone()
        target[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(batch, labels=target)
            nlls.append(outputs.loss.detach() * trg_len)

        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break

    avg_nll = torch.stack(nlls).sum() / seq_len
    ppl = torch.exp(avg_nll).item()
    if not math.isfinite(ppl):
        raise RuntimeError(f"Non-finite perplexity: {ppl}")

    return {
        "ppl": ppl,
        "seq_len": seq_len,
        "chunks": len(nlls),
    }


def build_publication_payload(
    *,
    label: str,
    base_result: dict[str, Any] | None,
    quantized_result: dict[str, Any] | None,
    quantized_pt: str | None,
    model_dir: str,
    wikitext: str,
    max_length: int,
    stride: int,
    token_limit: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": label,
        "timestamp_utc": int(time.time()),
        "model_dir": model_dir,
        "wikitext": wikitext,
        "quantized_pt": quantized_pt,
        "token_limit": "all" if token_limit is None else token_limit,
        "max_length": max_length,
        "stride": stride,
        "base": base_result,
        "quantized": None,
        "comparison": None,
    }

    if quantized_result is not None:
        compact_quantized = dict(quantized_result)
        compact_quantized["apply_stats"] = compact_apply_stats(
            quantized_result.get("apply_stats") or {}
        )
        payload["quantized"] = compact_quantized

    if base_result is not None and quantized_result is not None:
        base_ppl = float(base_result["ppl"])
        quantized_ppl = float(quantized_result["ppl"])
        payload["comparison"] = {
            "quantized_minus_base_ppl": quantized_ppl - base_ppl,
            "quantized_ppl_ratio_vs_base": quantized_ppl / base_ppl,
        }

    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate_apply_stats(args: argparse.Namespace, apply_stats: dict[str, Any]) -> None:
    compact = compact_apply_stats(apply_stats)
    if args.fail_on_missing and compact["missing_count"]:
        raise RuntimeError(f"Missing quantized model weights: {compact['missing_sample']}")
    if args.fail_on_shape_mismatch and compact["shape_mismatch_count"]:
        raise RuntimeError(
            f"Shape mismatches while applying checkpoint: {compact['shape_mismatch_sample']}"
        )
    if args.expect_replaced is not None and compact["replaced"] != args.expect_replaced:
        raise RuntimeError(
            f"Expected {args.expect_replaced} replaced weights, got {compact['replaced']}"
        )
    if (
        args.expect_skipped_shared_kv is not None
        and compact["skipped_count"] != args.expect_skipped_shared_kv
    ):
        raise RuntimeError(
            "Expected "
            f"{args.expect_skipped_shared_kv} skipped shared-KV entries, "
            f"got {compact['skipped_count']}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(args.model_dir)
    quantized_pt = Path(args.quantized_pt) if args.quantized_pt else None
    if quantized_pt is not None and not quantized_pt.exists():
        raise FileNotFoundError(f"Quantized checkpoint not found: {quantized_pt}")
    if not Path(args.wikitext).exists():
        raise FileNotFoundError(f"WikiText file not found: {args.wikitext}")

    token_limit = parse_token_limit(args.tokens)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    runtime_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    text = Path(args.wikitext).read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map=device,
        torch_dtype=runtime_dtype,
        trust_remote_code=True,
    )
    model.eval()

    base_result = None
    if not args.skip_base:
        start = time.time()
        base_result = eval_ppl(
            model,
            tokenizer,
            text,
            device,
            token_limit,
            args.max_length,
            args.stride,
        )
        base_result.update(
            {
                "elapsed_s": round(time.time() - start, 1),
                "device": device,
                "runtime_dtype": str(runtime_dtype),
            }
        )

    quantized_result = None
    checkpoint_stats = None
    if quantized_pt is not None:
        q_data = torch.load(quantized_pt, map_location="cpu", weights_only=True)
        checkpoint_stats = q_data.get("stats")
        apply_stats = apply_quantized_weights(
            model,
            q_data["quantized"],
            device=device,
            model_dir=model_dir,
            checkpoint_weight_keys=q_data.get("weight_keys"),
            strict=False,
        )
        validate_apply_stats(args, apply_stats)
        del q_data
        gc.collect()

        start = time.time()
        quantized_result = eval_ppl(
            model,
            tokenizer,
            text,
            device,
            token_limit,
            args.max_length,
            args.stride,
        )
        quantized_result.update(
            {
                "elapsed_s": round(time.time() - start, 1),
                "device": device,
                "runtime_dtype": str(runtime_dtype),
                "apply_stats": apply_stats,
                "checkpoint_stats": checkpoint_stats,
            }
        )

    payload = build_publication_payload(
        label=args.label,
        base_result=base_result,
        quantized_result=quantized_result,
        quantized_pt=str(quantized_pt) if quantized_pt is not None else None,
        model_dir=str(model_dir),
        wikitext=args.wikitext,
        max_length=args.max_length,
        stride=args.stride,
        token_limit=token_limit,
    )

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="groupwise_int4_g128_full")
    parser.add_argument("--model-dir", default="models/gemma-4-E2B")
    parser.add_argument("--wikitext", default="data/wiki.test.txt")
    parser.add_argument("--quantized-pt", default="quantized/gemma_groupwise_int4_g128.pt")
    parser.add_argument("--tokens", default="all", help="'all' for full WikiText, or an integer smoke limit")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--expect-replaced", type=int, default=None)
    parser.add_argument("--expect-skipped-shared-kv", type=int, default=None)
    parser.add_argument("--fail-on-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-shape-mismatch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="eval_results/groupwise_int4_g128_full_ppl.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    output = Path(args.output)
    write_json_atomic(output, payload)
    print("RESULT=" + json.dumps(payload, indent=2), flush=True)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
