import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import sys
from pathlib import Path

def reconstruct_weight(q_entry, device='cpu'):
    """Reconstruct weight from quantized format"""
    num_bits = q_entry['num_bits']
    q = q_entry['q'].to(device)
    scale = q_entry['scale']

    # Per-tensor dequantization
    qmax = 2 ** (num_bits - 1) - 1
    W = q.float() * scale / qmax
    return W

def eval_perplexity(model, tokenizer, wikitext_path: str, device: str = 'cpu', max_length: int = 512, stride: int = 128):
    with open(wikitext_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print("Tokenizing...")
    encodings = tokenizer(text, return_tensors='pt')
    encodings = {k: v.to(device) for k, v in encodings.items()}
    seq_len = encodings['input_ids'].shape[1]
    print(f"  Sequence length: {seq_len} tokens")

    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = encodings['input_ids'][:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc

        if end_loc >= seq_len:
            break

    avg_nll = torch.stack(nlls).sum() / seq_len
    perplexity = torch.exp(avg_nll).item()
    return perplexity, {'n_chunks': len(nlls), 'seq_len': seq_len}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantized-pt', default='quantized/gemma_hybrid_stream.pt')
    parser.add_argument('--model-dir', default='models/gemma-4-E2B')
    parser.add_argument('--wikitext', default='data/wiki.test.txt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--stride', type=int, default=128)
    args = parser.parse_args()

    device = args.device

    print("=" * 60)
    print("PERPLEXITY EVALUATION")
    print("=" * 60)

    # Load quantized
    print("\n[1] Loading quantized checkpoint...")
    q_data = torch.load(args.quantized_pt, map_location='cpu', weights_only=True)
    quantized = q_data['quantized']
    print(f"  {len(quantized)} quantized entries")
    print(f"  Method: {q_data.get('method', 'unknown')}")
    print(f"  Average bpw: {q_data['stats']['avg_bpw']:.4f}")

    # Load base model
    print("\n[2] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map=device,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()

    # Build mapping from shape to modules
    print("\n[3] Matching weights...")
    shape_to_modules = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            shape = tuple(module.weight.shape)
            if shape not in shape_to_modules:
                shape_to_modules[shape] = []
            shape_to_modules[shape].append((name, module))

    matched = 0
    for idx, q_entry in quantized.items():
        orig_shape = tuple(q_entry['shape'])
        if orig_shape in shape_to_modules:
            name, module = shape_to_modules[orig_shape][0]
            W_recon = reconstruct_weight(q_entry, device)
            with torch.no_grad():
                module.weight.data = W_recon.to(dtype=module.weight.dtype, device=module.weight.device)
            matched += 1

    print(f"  Matched {matched}/{len(quantized)} weights")

    # Evaluate perplexity
    print("\n[4] Evaluating perplexity...")
    wikitext_path = Path(args.wikitext)
    if not wikitext_path.exists():
        print(f"  WikiText not found at {wikitext_path}")
        print("  Run: python -c \"import urllib.request; urllib.request.urlretrieve('https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip', 'wikitext.zip'); import zipfile; zipfile.ZipFile('wikitext.zip').extractall('data/')\"")
        return

    ppl, stats = eval_perplexity(model, tokenizer, str(wikitext_path), device, max_length=args.max_length, stride=args.stride)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Perplexity: {ppl:.4f}")
    print(f"  Chunks: {stats['n_chunks']}")
    print(f"  Target: <= 10.5")
    status = "PASS" if ppl <= 10.5 else "FAIL"
    print(f"  Status: {status}")
    print("=" * 60)

    del model
    gc.collect()

if __name__ == "__main__":
    main()