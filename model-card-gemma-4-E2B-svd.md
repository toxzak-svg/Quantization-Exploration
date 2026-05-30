---
title: Gemma-4-E2B SVD Quantized
emoji: 🔬
colorFrom: purple
colorTo: blue
sdk: huggingface_hub
sdk_version: 0.26.0
app_file: app.py
pinned: false
tags:
- gemma
- quantization
- svd
- sub-1-bit
- pytorch
---

# Gemma-4-E2B SVD Quantized

SVD-based sub-1-bit quantization of Google's Gemma-4-E2B-IT model.

## Model Description

This repository contains SVD-quantized versions of [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it), achieved through singular value decomposition compression at 60% and 70% retention rates.

## Quantization Details

- **Base Model**: google/gemma-4-E2B-it (5B parameters)
- **Method**: SVD-based Sub-1-Bit Quantization
- **Variants**:
  - `gemma-4-E2B-svd60` - 60% singular value retention
  - `gemma-4-E2B-svd70` - 70% singular value retention

## Files

- `gemma-4-E2B-svd60.gguf` - GGUF format, 60% retention
- `gemma-4-E2B-svd70.gguf` - GGUF format, 70% retention
- `gemma-4-E2B-svd60/` - HuggingFace format, 60% retention
- `gemma-4-E2B-svd70/` - HuggingFace format, 70% retention

## Usage

### GGUF Format (llama.cpp, ollama, etc.)

```python
from huggingface_hub import hf_hub_download
download_path = hf_hub_download(repo_id="toxzak/gemma-4-E2B-svd-quantized", filename="gemma-4-E2B-svd60.gguf")
```

### HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("toxzak/gemma-4-E2B-svd-quantized", subfolder="gemma-4-E2B-svd60")
tokenizer = AutoTokenizer.from_pretrained("toxzak/gemma-4-E2B-svd-quantized", subfolder="gemma-4-E2B-svd60")
```

## Technical Notes

- Quantization performed using GPU acceleration
- Original model: https://huggingface.co/google/gemma-4-E2B-it
- License: Gemma Terms of Use

## Citation

If you use this model, please cite the original Gemma work and acknowledge this quantization effort.