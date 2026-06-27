"""
Text generation test using quantized weights.
Loads a base model, applies the quantized checkpoint, and generates text.
"""

import torch
import time
from pathlib import Path

from scripts.eval_quantized import apply_quantized_weights


def test_generation(
    model_name="models/gemma-4-E2B",
    quantized_path="quantized/gemma-4-E2B-sub1bit.pt",
    max_tokens=20,
):
    """Generate text after replacing model weights with a quantized checkpoint."""

    print(f"Loading quantized weights: {quantized_path}")
    q_data = torch.load(quantized_path, map_location='cpu', weights_only=True)
    quantized = q_data.get('quantized', {})
    print(f"Loaded {len(quantized)} quantized matrices")

    print(f"\nLoading tokenizer...")
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Tokenizer loaded")
    except Exception as e:
        print(f"Tokenizer error: {e}")
        return

    print(f"\nLoading model (will be slow, may OOM)...")
    try:
        # Load model with minimal memory
        from transformers import AutoModelForCausalLM

        # Try loading with low memory footprint
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.eval()
        print(f"Model loaded")

    except Exception as e:
        print(f"Model load error: {e}")
        print("Not enough RAM to load full model + quantized weights")
        return

    print(f"\nApplying quantized weights...")
    apply_stats = apply_quantized_weights(
        model,
        quantized,
        device="cpu",
        model_dir=model_name if Path(model_name).is_dir() else None,
        checkpoint_weight_keys=q_data.get('weight_keys'),
    )
    print(f"Replaced {apply_stats['replaced']}/{len(quantized)} weights")
    if apply_stats['skipped']:
        print(f"Skipped {len(apply_stats['skipped'])} shared-KV checkpoint entries")

    # Now generate
    print(f"\nGenerating text...")
    prompt = "Write a short poem about artificial intelligence:"

    inputs = tokenizer(prompt, return_tensors="pt")

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )

    elapsed = time.perf_counter() - start_time

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")
    print(f"Time: {elapsed:.2f}s for {max_tokens} tokens")
    print(f"Speed: {elapsed/max_tokens:.2f}s per token")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/gemma-4-E2B')
    parser.add_argument('--quantized', default='quantized/gemma-4-E2B-sub1bit.pt')
    parser.add_argument('--tokens', type=int, default=20)
    args = parser.parse_args()

    test_generation(args.model, args.quantized, args.tokens)
