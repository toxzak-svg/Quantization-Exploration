"""Verify model directory is complete and that we can read its config."""
import os, json
LOCAL = "/content/sub1quant/models/gemma-4-E2B"
required = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
    "generation_config.json",
    "processor_config.json",
]
for f in required:
    p = os.path.join(LOCAL, f)
    if os.path.exists(p):
        sz = os.path.getsize(p)
        print(f"  OK  {f}  {sz:,} bytes ({sz/1e6:.1f} MB)")
    else:
        print(f"  MISS  {f}")

# Show config summary
cfg = json.load(open(os.path.join(LOCAL, "config.json")))
print("model_type:", cfg.get("model_type"))
print("text hidden_size:", cfg["text_config"]["hidden_size"])
print("text layers:", cfg["text_config"]["num_hidden_layers"])

# Check the existing quantized dir exists
qd = "/content/sub1quant/quantized"
print(f"\nquantized dir: {qd}")
if os.path.isdir(qd):
    for f in sorted(os.listdir(qd)):
        full = os.path.join(qd, f)
        if os.path.isfile(full):
            print(f"  {f}  {os.path.getsize(full):,}")
        else:
            print(f"  {f}/")
else:
    print("  not present yet")

# Also check the .pt upload progress (any file with that name?)
target = "/content/sub1quant/quantized/gemma_mixed_budget_full_g128_target4p0.pt"
if os.path.exists(target):
    sz = os.path.getsize(target)
    print(f"\n.pt current size: {sz:,} bytes ({sz/1024/1024:.1f} MB)")
else:
    print(f"\n.pt not yet at {target}")
