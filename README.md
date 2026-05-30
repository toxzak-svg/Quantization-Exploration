# Sub1BitLLM Colab Package

Ultra-low bit quantization (≤0.7 bits/weight) for LLM inference.

## Quick Start

1. Upload `sub1quant_colab.zip` to Google Drive
2. In Colab: `!unzip sub1quant_colab.zip -d /content/`
3. Open `Sub1BitLLM_Debug.ipynb` and run all cells

## API Usage

```python
import sys
sys.path.insert(0, '/content/sub1quant/src')

from Sub1BitLLM import Sub1BitLLM, Sub1BitConfig, from_fp16

# Load from FP16 checkpoints
model = from_fp16(
    "/content/models/llama-2-7b",
    config=Sub1BitConfig(energy_threshold=0.95),
    checkpoint_dir="/content/sub1quant/checkpoints/"
)

# Memory-efficient inference
for idx, layer in model.iter_layers():
    output = layer.forward_lowrank(x)  # x @ W without materializing W

# Export to GGUF
model.to_gguf("/content/sub1quant/model.gguf")
```

## Project Structure

```
src/
├── Sub1BitLLM.py           # Core API: Sub1BitLLM, Sub1BitConfig, from_fp16
├── train_transform.py      # Learnable transform + binary codebook
├── lowrank_factorization.py # SVD decomposition
├── quantize.py             # Full quantization pipeline
├── pack_gguf.py            # GGUF export
└── __init__.py             # Unified exports + Sub1BitPipeline
```

## Files

- `Sub1BitLLM_Debug.ipynb` - Colab notebook with test cells
- `README.md` - This file

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
transformers>=4.30.0
```

## Target Metrics

| Metric | Target |
|--------|--------|
| Average bit-width | ≤0.7 bit/weight |
| WikiText-2 perplexity | ≤10.5 |
| Zero-shot accuracy drop | ≤2% |