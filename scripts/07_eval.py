#!/usr/bin/env python3
import subprocess
import argparse
import os
import re
import torch


def get_gguf_size(filepath: str) -> int:
    return os.path.getsize(filepath)


def run_perplexity(gguf_path: str, wikitext_path: str, llama_cpp_path: str) -> float:
    perplexity_exe = os.path.join(llama_cpp_path, "build", "bin", "llama-perplexity.exe")

    if not os.path.exists(perplexity_exe):
        build_cmd = f'cmake --build build --config Release'
        print(f"Building llama.cpp first...")
        return None

    cmd = [
        perplexity_exe,
        "-m", gguf_path,
        "-f", wikitext_path,
        "-t", "8"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr

        ppl_match = re.search(r"perplexity:\s+([0-9.]+)", output)
        if ppl_match:
            return float(ppl_match.group(1))
    except Exception as e:
        print(f"Error running perplexity: {e}")

    return None


def estimate_bit_width(gguf_size_bytes: int, num_parameters: int = 7_000_000_000) -> float:
    total_bits = gguf_size_bytes * 8
    return total_bits / num_parameters


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized model")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model")
    parser.add_argument("--wikitext", type=str, default="data/wiki.test.txt", help="WikiText test file")
    parser.add_argument("--llama-cpp", type=str, default="llama.cpp", help="llama.cpp directory")
    parser.add_argument("--task", choices=["all", "perplexity", "size"], default="all")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        return

    print("=" * 60)
    print("Sub-1-Bit Quantization Evaluation")
    print("=" * 60)

    if args.task in ["all", "size"]:
        size = get_gguf_size(args.model)
        bit_width = estimate_bit_width(size)

        print(f"\n[Size Analysis]")
        print(f"  GGUF size: {size / 1024 / 1024:.2f} MB")
        print(f"  Estimated bit-width: {bit_width:.4f} bits/weight")

        if bit_width <= 0.7:
            print(f"  Status: PASS (target: ≤0.7 bits/weight)")
        else:
            print(f"  Status: FAIL (target: ≤0.7 bits/weight)")

    if args.task in ["all", "perplexity"]:
        print(f"\n[Perplexity Evaluation]")
        ppl = run_perplexity(args.model, args.wikitext, args.llama_cpp)

        if ppl is not None:
            print(f"  WikiText-2 PPL: {ppl:.4f}")
            if ppl <= 10.5:
                print(f"  Status: PASS (target: ≤10.5)")
            else:
                print(f"  Status: FAIL (target: ≤10.5)")
        else:
            print("  Could not run perplexity evaluation")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()