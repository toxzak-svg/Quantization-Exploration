import torch
import gc
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.quantization import ternary_unpack, sigma_dequantize


def reconstruct_weight(q_entry: dict, device: str = 'cpu') -> torch.Tensor:
    U = ternary_unpack(q_entry['U_packed'].to(device), q_entry['U_shape']).float() * q_entry['U_scale'].to(device).float()
    Vt = ternary_unpack(q_entry['Vt_packed'].to(device), q_entry['Vt_shape']).float() * q_entry['Vt_scale'].to(device).float()
    S = sigma_dequantize(q_entry['S'].to(device), q_entry['S_scale'].to(device))
    return torch.matmul(U * S.unsqueeze(0), Vt)


def eval_perplexity(model, tokenizer, wikitext_path: str, device: str,
                    max_length: int = 512, stride: int = 512):
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
    parser = argparse.ArgumentParser(description="Evaluate quantized sub-1bit model perplexity")
    parser.add_argument('--quantized-pt', default='llama-2-7b-sub1bit.gguf.pt',
                        help='Path to quantized .pt checkpoint')
    parser.add_argument('--model-dir', default='models/llama-2-7b-hf',
                        help='Path to base model directory')
    parser.add_argument('--wikitext', default='data/wiki.test.txt',
                        help='Path to WikiText test file')
    parser.add_argument('--device', default=None,
                        help='Device (auto-detect if not set)')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--stride', type=int, default=512)
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    wikitext_path = Path(args.wikitext)
    if not wikitext_path.exists():
        print(f"WikiText not found: {wikitext_path}")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("QUANTIZED MODEL PERPLEXITY EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Quantized: {args.quantized_pt}")
    print(f"Base model: {args.model_dir}")
    print(f"WikiText: {args.wikitext}")
    print()

    # 1. Load quantized checkpoint
    print("[1] Loading quantized checkpoint...")
    q_data = torch.load(args.quantized_pt, map_location='cpu', weights_only=True)
    quantized = q_data['quantized']
    print(f"  {len(quantized)} quantized entries")
    print()

    # 2. Load base model
    print("[2] Loading base model...")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map=device,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    model.eval()
    print()

    # 3. Reconstruct weights
    print("[3] Reconstructing quantized weights...")
    name_to_idx = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and name.endswith('.weight'):
            name_to_idx[name.replace('.weight', '')] = module

    matched_entries = 0
    for q_idx_str, q_entry in quantized.items():
        orig_shape = tuple(q_entry['original_shape'])
        for mod_name, module in name_to_idx.items():
            if tuple(module.weight.shape) == orig_shape:
                W_recon = reconstruct_weight(q_entry, device)
                with torch.no_grad():
                    module.weight.data = W_recon.to(
                        dtype=module.weight.dtype, device=module.weight.device
                    )
                matched_entries += 1
                break

    print(f"  Replaced {matched_entries}/{len(quantized)} weights")
    print()

    # 4. Evaluate perplexity
    print("[4] Evaluating perplexity...")
    ppl, stats = eval_perplexity(
        model, tokenizer, str(wikitext_path), device,
        max_length=args.max_length, stride=args.stride
    )
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
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
