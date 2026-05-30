---
library_name: transformers
license: gemma
license_link: https://ai.google.dev/gemma/terms
pipeline_tag: text-generation
---

<div align="center">
  <img src=https://ai.google.dev/gemma/images/gemma4_banner.png>
</div>


<p align="center">
    <a href="https://huggingface.co/collections/google/gemma-2-2b-release" target="_blank">Hugging Face</a> |
    <a href="https://github.com/google-gemma" target="_blank">GitHub</a> |
    <a href="https://ai.google.dev/gemma/docs/core" target="_blank">Documentation</a>
    <br>
    <b>License</b>: <a href="https://ai.google.dev/gemma/terms" target="_blank">Gemma Terms</a> | <b>Authors</b>: <a href="https://deepmind.google/models/gemma/" target="_blank">Google DeepMind</a>
</p>

Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. Gemma 2 is the second generation, featuring improved architecture with sliding window attention, attention soft-capping, and optimized feed-forward networks. These models are well-suited for a variety of text generation tasks including question answering, summarization, and reasoning.

## **Model Overview**

Gemma 2 models are decoder-only transformer text models available in multiple sizes. The 2B model is designed for efficient deployment in resource-constrained environments while maintaining strong performance across benchmarks.

### **Architecture Details**

| Property | Value |
| :---- | :---- |
| **Total Parameters** | ~2.6B |
| **Layers** | 26 |
| **Hidden Size** | 2048 |
| **Head Dimension** | 256 |
| **Attention Heads** | 8 |
| **Key Value Heads** | 1 (Grouped Query Attention) |
| **Sliding Window** | 4096 tokens |
| **Context Length** | 8K tokens |
| **Vocabulary Size** | ~256K |
| **Attention Soft Cap** | 50.0 |
| **Final Logit Soft Cap** | 30.0 |

Key architectural features:
- **Sliding Window Attention**: Alternating layers with 4096-token sliding window and full attention for long-range dependencies
- **Attention Soft-Capping**: Prevents attention logits from exceeding reasonable bounds
- **Final Logit Soft-Capping**: Applied to output logits for stable generation
- **Grouped Query Attention**: Efficient KV cache with shared query projections

## **Benchmark Results**

Benchmark results for base pre-trained models:

| Benchmark | Metric | Gemma 2 2B | Gemma 2 9B | Gemma 2 27B |
| :---- | :---- | :---- | :---- | :---- |
| MMLU | 5-shot, top-1 | 51.3% | 71.3% | 75.2% |
| HellaSwag | 10-shot | 73.0% | 81.9% | 86.4% |
| PIQA | 0-shot | 77.8% | 81.7% | 83.2% |
| SocialIQA | 0-shot | 51.9% | 53.4% | 53.7% |
| BoolQ | 0-shot | 72.5% | 84.2% | 84.8% |
| WinoGrande | partial score | 70.9% | 80.6% | 83.7% |
| ARC-e | 0-shot | 80.1% | 88.0% | 88.6% |
| ARC-c | 25-shot | 55.4% | 68.4% | 71.4% |
| TriviaQA | 5-shot | 59.4% | 76.6% | 83.7% |
| Natural Questions | 5-shot | 16.7% | 29.2% | 34.5% |
| HumanEval | pass@1 | 17.7% | 40.2% | 51.8% |
| MBPP | 3-shot | 29.6% | 52.4% | 62.6% |
| GSM8K | 5-shot, maj@1 | 23.9% | 68.6% | 74.0% |
| MATH | 4-shot | 15.0% | 36.6% | 42.3% |
| AGIEval | 3-5-shot | 30.6% | 52.8% | 55.1% |
| DROP | 3-shot, F1 | 52.0% | 69.4% | 72.2% |
| BIG-Bench | 3-shot, CoT | 41.9% | 68.2% | 74.9% |

## **Usage**

### Installation

```bash
pip install -U transformers torch accelerate
```

### Running with pipeline

```python
import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
]

outputs = pipe(messages, max_new_tokens=256)
print(outputs[0]["generated_text"][-1]["content"])
```

### Running on single/multi GPU

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

messages = [
    {"role": "user", "content": "Write a hello world program"},
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

### Quantized versions via bitsandbytes

```python
# 8-bit precision
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    quantization_config=quantization_config,
)
```

```python
# 4-bit precision
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    quantization_config=quantization_config,
)
```

### Chat Template

The instruction-tuned model uses a chat template with `<start_of_turn>` and `<end_of_turn>` delimiters:

```
<bos><start_of_turn>user
Write a hello world program<end_of_turn>
<start_of_turn>model
```

## **Model Data**

### Training Dataset

These models were trained on a dataset of text data including:
- **Web Documents**: Diverse English-language web text
- **Code**: Programming language syntax and patterns
- **Mathematics**: Logical reasoning and symbolic representation

Training tokens: 2 trillion tokens for 2B model.

### Data Preprocessing

- **CSAM Filtering**: Rigorous CSAM filtering at multiple stages
- **Sensitive Data Filtering**: Automated techniques to filter personal information
- **Content Quality Filtering**: Based on Google policies

## **Implementation Information**

### Hardware

Trained using TPUv5p hardware (TPU = Tensor Processing Unit).

### Software

Training done using JAX and ML Pathways.

## **Ethics and Safety**

### Evaluation Approach

Evaluations include structured evaluations and internal red-teaming:
- Text-to-Text Content Safety
- Text-to-Text Representational Harms
- Memorization testing
- Dangerous capabilities testing (CBRN risks)

### Safety Benchmark Results

| Benchmark | Metric | Gemma 2 IT 2B | Gemma 2 IT 9B | Gemma 2 IT 27B |
| :---- | :---- | :---- | :---- | :---- |
| RealToxicity | average | 8.16 | 8.25 | 8.84 |
| CrowS-Pairs | top-1 | 37.67 | 37.47 | 36.67 |
| BBQ Ambig | 1-shot, top-1 | 83.20 | 88.58 | 85.99 |
| BBQ Disambig | top-1 | 69.31 | 82.67 | 86.94 |
| Winogender | top-1 | 52.91 | 79.17 | 77.22 |
| TruthfulQA | - | 43.72 | 50.27 | 51.60 |
| Winobias 1_2 | - | 59.28 | 78.09 | 81.94 |
| Winobias 2_2 | - | 88.57 | 95.32 | 97.22 |
| ToxiGen | - | 48.32 | 39.30 | 38.42 |

## **Usage and Limitations**

### Intended Usage

- **Content Creation**: Creative text, code, summaries
- **Chatbots**: Conversational AI for customer service and virtual assistants
- **Research**: NLP experimentation and language learning tools

### Limitations

- Training data quality affects model capabilities
- LLMs struggle with open-ended or highly complex tasks
- Natural language ambiguity and nuance
- Factual accuracy depends on training data
- Common sense reasoning may be lacking

### Prohibited Uses

Prohibited uses of Gemma models are outlined in the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

## **Citation**

```none
@article{gemma_2024,
    title={Gemma},
    url={https://www.kaggle.com/m/3301},
    DOI={10.34740/KAGGLE/M/3301},
    publisher={Kaggle},
    author={Gemma Team},
    year={2024}
}
```