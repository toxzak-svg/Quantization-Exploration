import torch, gc, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from eval_quantized import apply_quantized_weights, eval_perplexity

device = 'cpu'
print("=" * 60)
print("QUANTIZED MODEL PERPLEXITY EVALUATION")
print("=" * 60)
print(f"Device: {device}")

print("\n[1] Loading quantized checkpoint...")
q_data = torch.load('quantized/gemma-4-E2B-sub1bit.pt', map_location='cpu', weights_only=True)
quantized = q_data['quantized']
print(f"  {len(quantized)} quantized entries")

print("\n[2] Loading base model...")
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('models/gemma-4-E2B', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    'models/gemma-4-E2B',
    device_map=device,
    torch_dtype=torch.float32,
    trust_remote_code=True
)
model.eval()
print("  Model loaded")

print("\n[3] Reconstructing quantized weights...")
apply_stats = apply_quantized_weights(
    model,
    quantized,
    device=device,
    model_dir='models/gemma-4-E2B',
    checkpoint_weight_keys=q_data.get('weight_keys'),
)
print(f"  Replaced {apply_stats['replaced']}/{len(quantized)} weights")
if apply_stats['skipped']:
    print(f"  Skipped {len(apply_stats['skipped'])} shared-KV checkpoint entries")

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
