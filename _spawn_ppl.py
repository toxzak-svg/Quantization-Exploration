"""Launch the full perplexity eval as a background subprocess on the kernel,
and write progress to /content/sub1quant/_ppl.log + _ppl_state.json
"""
import os, subprocess, json, time
LOG = "/content/sub1quant/_ppl.log"
SCRIPT = "/content/sub1quant/test_perplexity.py"
os.makedirs("/content/sub1quant/eval_results", exist_ok=True)

cmd = (
    f"nohup python -u {SCRIPT} "
    f"--model /content/sub1quant/models/gemma-4-E2B "
    f"--quantized /content/sub1quant/quantized/gemma_mixed_budget_full_g128_target4p0.pt "
    f"--wikitext /content/sub1quant/data/wiki.test.txt "
    f"--device cuda "
    f"--max-length 512 --stride 512 "
    f"> {LOG} 2>&1 & echo $!"
)
out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
pid = out.stdout.strip().split("\n")[-1]
print(f"OK pid={pid}")
print(f"LOG={LOG}")
