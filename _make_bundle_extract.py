"""Extract the eval bundle to /content/ and verify all expected files are there.
Run this on the colab kernel via /exec."""
import sys, os, tarfile, pathlib
BUNDLE = "/tmp/_eval_bundle.tar.gz"
TARGET = pathlib.Path("/content")

ok = True
errs = []
if not os.path.exists(BUNDLE):
    errs.append(f"missing bundle: {BUNDLE}")
else:
    tarfile.open(BUNDLE, "r:gz").extractall(TARGET)

required = [
    "/content/sub1quant/scripts/eval_quantized.py",
    "/content/sub1quant/scripts/quantize_mixed_budget.py",
    "/content/sub1quant/scripts/limited_ppl_bench.py",
    "/content/sub1quant/test_perplexity.py",
    "/content/sub1quant/src/__init__.py",
    "/content/sub1quant/src/error_budget_residual.py",
    "/content/sub1quant/src/groupwise_int4.py",
    "/content/sub1quant/src/quantization.py",
    "/content/sub1quant/src/mixed_budget.py",
    "/content/sub1quant/src/lowrank_factorization.py",
    "/content/sub1quant/data/wiki.test.txt",
]
for p in required:
    if not os.path.exists(p):
        errs.append(f"MISSING: {p}")
        ok = False
    else:
        print(f"OK  : {p}  ({os.path.getsize(p)} bytes)")

# Make sure src dir is on PYTHONPATH for the project's modules
os.environ["PYTHONPATH"] = "/content/sub1quant:" + os.environ.get("PYTHONPATH", "")
print("PYTHONPATH=", os.environ["PYTHONPATH"])

# Quick sanity import
sys.path.insert(0, "/content/sub1quant")
try:
    import scripts.eval_quantized as eq  # noqa: F401
    print("imported scripts.eval_quantized OK")
except Exception as e:
    errs.append(f"import eval_quantized FAILED: {e}")
    ok = False

try:
    from src import error_budget_residual, groupwise_int4, quantization, mixed_budget  # noqa: F401
    print("imported src.* OK")
except Exception as e:
    errs.append(f"import src.* FAILED: {e}")
    ok = False

if errs:
    print("ERRORS:", *errs, sep="\n  ")
print("READY" if ok else "FAIL")
