"""Verify the eval setup is importable. Run on the colab kernel via /exec."""
import sys, os, traceback
os.environ["PYTHONPATH"] = "/content/sub1quant:" + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, "/content/sub1quant")

results = {}
def try_import(label, fn):
    try:
        fn()
        results[label] = "OK"
        print(f"OK   {label}")
    except Exception as e:
        results[label] = f"FAIL: {e}"
        print(f"FAIL {label}: {e}")

# Direct submodule imports (NOT through src/__init__.py) — that's what eval_quantized.py uses
def t1():
    from src import error_budget_residual
    from src import groupwise_int4
    from src import quantization
    from src import mixed_budget
def t2():
    import scripts.eval_quantized as eq
    # Check key functions exist
    assert callable(eq.eval_perplexity)
    assert callable(eq.apply_quantized_weights)
def t3():
    # Verify torch + cuda
    import torch
    assert torch.cuda.is_available()
    print(f"  cuda device: {torch.cuda.get_device_name(0)}")

def t4():
    # Verify wikitext is loadable
    with open("/content/sub1quant/data/wiki.test.txt", "r", encoding="utf-8") as f:
        s = f.read()
    n = len(s)
    print(f"  wiki chars: {n}")
    assert n > 1_000_000, f"wikitext too short: {n}"

try_import("submodule direct imports", t1)
try_import("scripts.eval_quantized", t2)
try_import("torch+cuda", t3)
try_import("wikitext loadable", t4)

print("ALL" if all(v == "OK" for v in results.values()) else "PARTIAL")
