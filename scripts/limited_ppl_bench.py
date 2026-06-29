"""Run a short WikiText perplexity smoke benchmark for base/quantized checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_quantized import apply_quantized_weights


def eval_limited_ppl(
    model,
    tokenizer,
    text: str,
    device: str,
    tokens: int,
    max_length: int,
    stride: int,
) -> dict:
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"][:, :tokens].to(device)
    seq_len = input_ids.shape[1]
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

    ppl = torch.exp(torch.stack(nlls).sum() / seq_len).item()
    return {"ppl": ppl, "seq_len": seq_len, "chunks": len(nlls)}


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(args.model_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    apply_stats = {"replaced": 0, "skipped": []}
    checkpoint_stats = None
    if args.quantized_pt:
        q_data = torch.load(args.quantized_pt, map_location="cpu", weights_only=True)
        checkpoint_stats = q_data.get("stats")
        apply_stats = apply_quantized_weights(
            model,
            q_data["quantized"],
            device=device,
            model_dir=model_dir,
            checkpoint_weight_keys=q_data.get("weight_keys"),
            strict=False,
        )
        del q_data
        gc.collect()

    text = Path(args.wikitext).read_text(encoding="utf-8")
    metrics = eval_limited_ppl(
        model,
        tokenizer,
        text,
        device,
        tokens=args.tokens,
        max_length=args.max_length,
        stride=args.stride,
    )
    metrics.update(
        {
            "label": args.label,
            "mode": "quantized" if args.quantized_pt else "base",
            "quantized_pt": args.quantized_pt,
            "apply_stats": apply_stats,
            "checkpoint_stats": checkpoint_stats,
            "device": device,
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-dir", default="models/gemma-4-E2B")
    parser.add_argument("--wikitext", default="data/wiki.test.txt")
    parser.add_argument("--quantized-pt", default=None)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    result = run(args)
    result["elapsed_s"] = round(time.time() - start, 1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("RESULT=" + json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
