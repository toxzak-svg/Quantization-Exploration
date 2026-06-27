"""
Perplexity evaluation for quantized .pt checkpoints.

This runs a real transformer forward pass after applying quantized weights.
For GGUF/GEMV latency checks, use run_gemv_chain.py instead.
"""

import argparse
import gc
from pathlib import Path

import torch

from scripts.eval_quantized import apply_quantized_weights, eval_perplexity


def evaluate_quantized_perplexity(
    model_name="models/gemma-4-E2B",
    quantized_path="quantized/gemma-4-E2B-sub1bit.pt",
    wikitext_path="data/wiki.test.txt",
    device=None,
    max_length=512,
    stride=512,
):
    """Evaluate WikiText perplexity after applying quantized weights."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(wikitext_path).exists():
        raise FileNotFoundError(f"WikiText not found: {wikitext_path}")

    print("=" * 60)
    print("QUANTIZED MODEL PERPLEXITY EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Quantized: {quantized_path}")
    print(f"WikiText: {wikitext_path}")

    print("\n[1] Loading quantized checkpoint...")
    q_data = torch.load(quantized_path, map_location="cpu", weights_only=True)
    quantized = q_data["quantized"]
    print(f"  {len(quantized)} quantized entries")

    print("\n[2] Loading base model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()

    print("\n[3] Applying quantized weights...")
    apply_stats = apply_quantized_weights(
        model,
        quantized,
        device=device,
        model_dir=model_name if Path(model_name).is_dir() else None,
        checkpoint_weight_keys=q_data.get("weight_keys"),
    )
    print(f"  Replaced {apply_stats['replaced']}/{len(quantized)} weights")
    if apply_stats["skipped"]:
        print(f"  Skipped {len(apply_stats['skipped'])} shared-KV checkpoint entries")

    print("\n[4] Evaluating perplexity...")
    ppl, stats = eval_perplexity(
        model,
        tokenizer,
        wikitext_path,
        device,
        max_length=max_length,
        stride=stride,
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Perplexity: {ppl:.4f}")
    print(f"  Chunks: {stats['n_chunks']}")
    print(f"  Target: <= 10.5")
    print(f"  Status: {'PASS' if ppl <= 10.5 else 'FAIL'}")
    print("=" * 60)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return ppl, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate quantized model perplexity")
    parser.add_argument("--model", default="models/gemma-4-E2B")
    parser.add_argument("--quantized", default="quantized/gemma-4-E2B-sub1bit.pt")
    parser.add_argument("--wikitext", default="data/wiki.test.txt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    args = parser.parse_args()

    evaluate_quantized_perplexity(
        model_name=args.model,
        quantized_path=args.quantized,
        wikitext_path=args.wikitext,
        device=args.device,
        max_length=args.max_length,
        stride=args.stride,
    )
