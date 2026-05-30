import torch, gc, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from eval_quantized import eval_perplexity, reconstruct_weight

device = 'cpu'
print("=" * 60)
print("QUANTIZED MODEL PERPLEXITY EVALUATION")
print("=" * 60)
print(f"Device: {device}")

print("\n[1] Loading quantized checkpoint...")
q_data = torch.load('llama-2-7b-sub1bit.gguf.pt', map_location='cpu', weights_only=True)
quantized = q_data['quantized']
print(f"  {len(quantized)} quantized entries")

print("\n[2] Loading base model...")
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    device_map=device,
    torch_dtype=torch.float32,
    trust_remote_code=True
)
model.eval()
print("  Model loaded from HuggingFace")

print("\n[3] Reconstructing quantized weights...")
name_to_idx = {}
for name, module in model.named_modules():
    if hasattr(module, 'weight') and name.endswith('.weight'):
        name_to_idx[name.replace('.weight', '')] = module

matched = 0
for q_idx_str, q_entry in quantized.items():
    orig_shape = tuple(q_entry['original_shape'])
    for mod_name, module in name_to_idx.items():
        if tuple(module.weight.shape) == orig_shape:
            W_recon = reconstruct_weight(q_entry, device)
            with torch.no_grad():
                module.weight.data = W_recon.to(dtype=module.weight.dtype, device=module.weight.device)
            matched += 1
            break
print(f"  Replaced {matched}/{len(quantized)} weights")

print("\n[4] Evaluating perplexity...")
ppl, stats = eval_perplexity(model, tokenizer, 'data/wiki.test.txt', device)
print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  Perplexity: {ppl:.4f}")
print(f"  Chunks: {stats['n_chunks']}")
print(f"  Status: {'PASS' if ppl <= 10.5 else 'FAIL'} (target <= 10.5)")
print("=" * 60)

del model; gc.collect()
