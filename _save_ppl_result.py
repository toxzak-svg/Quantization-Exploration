"""Save the ppl result as JSON for posterity."""
import json, time, re, os, sys
LOG = "/content/sub1quant/_ppl.log"
PT = "/content/sub1quant/quantized/gemma_mixed_budget_full_g128_target4p0.pt"
OUT = "/content/sub1quant/eval_results/mixed_budget_full_g128_target4p0_ppl_colab.json"

# Parse what we got
log = open(LOG).read()
m_ppl = re.search(r"Perplexity:\s+([\d.]+)", log)
m_chunks = re.search(r"Chunks:\s+(\d+)", log)
m_seq = re.search(r"Sequence length:\s+(\d+)", log)
m_replaced = re.search(r"Replaced\s+(\d+)/(\d+)\s+weights", log)
m_skipped = re.search(r"Skipped\s+(\d+)\s+shared-KV", log)

result = {
    "timestamp_utc": int(time.time()),
    "method": "mixed_budget",
    "group_size": 128,
    "target_bpw": 4.0,
    "quantized_pt": PT,
    "model_dir": "/content/sub1quant/models/gemma-4-E2B",
    "wikitext": "/content/sub1quant/data/wiki.test.txt",
    "device": "cuda",
    "max_length": 512,
    "stride": 512,
    "seq_len_tokens": int(m_seq.group(1)) if m_seq else None,
    "n_chunks": int(m_chunks.group(1)) if m_chunks else None,
    "weights_replaced": int(m_replaced.group(1)) if m_replaced else None,
    "weights_total_pt": int(m_replaced.group(2)) if m_replaced else None,
    "weights_skipped_shared_kv": int(m_skipped.group(1)) if m_skipped else None,
    "perplexity": float(m_ppl.group(1)) if m_ppl else None,
    "target_ppl": 10.5,
    "status": "FAIL" if (m_ppl and float(m_ppl.group(1)) > 10.5) else "PASS",
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print("wrote", OUT)
print(json.dumps(result, indent=2))
