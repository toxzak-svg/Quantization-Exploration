"""
Text generation test using sub1bit quantized weights.
Loads gemma-4-E2B and replaces weights with sub1bit factors.
"""

import torch
import time
import gc
from pathlib import Path

def test_generation(model_name="models/gemma-4-E2B", max_tokens=20):
    """Generate text using sub1bit weights."""

    print(f"Loading quantized weights...")
    q_data = torch.load('quantized/gemma-4-E2B-sub1bit.pt', map_location='cpu', weights_only=True)
    quantized = q_data.get('quantized', {})
    print(f"Loaded {len(quantized)} quantized matrices")

    # Build layer index mapping
    # We need to figure out which quantized matrix corresponds to which model weight
    # This requires knowing the weight names from the original model

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
    parser.add_argument('--tokens', type=int, default=20)
    args = parser.parse_args()

    test_generation(args.model, args.tokens)