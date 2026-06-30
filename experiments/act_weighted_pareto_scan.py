"""Activation-weighted mixed-budget Pareto scan for sub1quant.

This driver runs the full Pareto-front experiment end-to-end:

  1. Collect per-layer activation stats from a forward pass over WikiText.
  2. Run scan_mixed_budget at multiple target_bpw values using those stats.
  3. Build the chosen mixed-budget checkpoint for each target.
  4. Run full WikiText perplexity on each checkpoint.
  5. Write a summary JSON + human-readable report.

The driver is idempotent and resumable: each stage checks for its output and
skips if present. Re-runs continue from where they left off.

Designed to run on a single Colab L4 runtime. Drive paths are used for any
output that needs to survive runtime reset / disconnect.

Usage on Colab (after cloning the repo and installing deps):

    cd /content/sub1quant
    python3 -u experiments/act_weighted_pareto_scan.py 2>&1 | tee /content/run.log

Poll from Windows:

    # from the colab bridge
    exec:  !tail -n 50 /content/run.log
    exec:  !ls /content/sub1quant/eval_results/act_weighted_scan/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "gemma-4-E2B"
DEFAULT_WIKITEXT = PROJECT_ROOT / "data" / "wiki.test.txt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results" / "act_weighted_scan"

DEFAULT_TARGETS_BPW = [3.75, 3.5, 3.25, 3.0, 2.75, 2.5]
DEFAULT_GROUP_SIZE = 128
DEFAULT_OUTLIER_OPTIONS = [4, 8]
DEFAULT_ACT_TOKENS = 32_768
DEFAULT_ACT_MAX_LENGTH = 512
DEFAULT_ACT_STRIDE = 512
PPL_TOKENS_CAP = 1_000_000_000  # full WikiText
PPL_MAX_LENGTH = 512
PPL_STRIDE = 512


def run_subprocess(cmd: list[str], log_path: Path | None = None) -> int:
    """Run a subprocess, streaming output to stdout and optionally a log file."""
    print(f"$ {' '.join(cmd)}", flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as logf:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
                logf.flush()
            rc = proc.wait()
    else:
        rc = subprocess.call(cmd)
    return rc


def ensure_activation_weights(
    out_dir: Path,
    model_dir: Path,
    wikitext: Path,
    act_tokens: int,
    max_length: int,
    stride: int,
    device: str,
) -> Path:
    """Run collect_activation_weights.py if activation_weights_gemma4.json is missing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "activation_weights_gemma4.json"
    progress = out_dir / "activation_weights_progress.json"

    if target.exists() and json.loads(target.read_text(encoding="utf-8")):
        print(f"[act] Activation weights already exist: {target}")
        return target

    if progress.exists():
        progress_data = json.loads(progress.read_text(encoding="utf-8"))
        if progress_data.get("complete"):
            print(f"[act] Activation weights complete per progress file: {target}")
            return target

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "collect_activation_weights.py"),
        "--model-dir", str(model_dir),
        "--wikitext", str(wikitext),
        "--output", str(target),
        "--stats-output", str(out_dir / "activation_stats_gemma4.json"),
        "--progress-output", str(progress),
        "--tokens", str(act_tokens),
        "--max-length", str(max_length),
        "--stride", str(stride),
        "--save-every-chunks", "1",
        "--device", device,
    ]
    log = out_dir / "logs" / "collect.log"
    rc = run_subprocess(cmd, log_path=log)
    if rc != 0:
        raise RuntimeError(f"collect_activation_weights.py exited with {rc}")
    if not target.exists():
        raise RuntimeError(f"collect did not produce {target}")
    return target


def run_scan(
    target_bpw: float,
    out_dir: Path,
    model_dir: Path,
    activation_weights: Path,
    group_size: int,
    outlier_options: list[int],
) -> Path:
    """Run scan_mixed_budget.py at the given target_bpw using the activation weights."""
    scan_path = out_dir / "scans" / f"target_{target_bpw:.2f}.json"
    resume_path = out_dir / "scans" / f"target_{target_bpw:.2f}.layers.jsonl"

    if scan_path.exists():
        data = json.loads(scan_path.read_text(encoding="utf-8"))
        if data.get("mixed_allocation", {}).get("selected_layers"):
            print(f"[scan {target_bpw}] scan already exists: {scan_path}")
            return scan_path

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "scan_mixed_budget.py"),
        "--model-dir", str(model_dir),
        "--group-size", str(group_size),
        "--outlier-options", ",".join(str(k) for k in outlier_options),
        "--target-bpw", str(target_bpw),
        "--activation-weights", str(activation_weights),
        "--resume-jsonl", str(resume_path),
        "--output", str(scan_path),
    ]
    log = out_dir / "logs" / f"scan_{target_bpw:.2f}.log"
    rc = run_subprocess(cmd, log_path=log)
    if rc != 0:
        raise RuntimeError(f"scan_mixed_budget.py target={target_bpw} exited with {rc}")
    return scan_path


def run_build(
    target_bpw: float,
    scan_path: Path,
    out_dir: Path,
    model_dir: Path,
    group_size: int,
) -> Path:
    """Run quantize_mixed_budget.py to build the checkpoint for this target."""
    output = out_dir / "checkpoints" / f"gemma_mixed_act_{target_bpw:.2f}.pt"
    shard_dir = out_dir / "shards" / f"target_{target_bpw:.2f}"

    if output.exists() and output.stat().st_size > 100_000_000:
        print(f"[build {target_bpw}] checkpoint already exists: {output}")
        return output

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "quantize_mixed_budget.py"),
        "--scan-json", str(scan_path),
        "--model-dir", str(model_dir),
        "--group-size", str(group_size),
        "--checkpoint-dir", str(shard_dir),
        "--output", str(output),
    ]
    log = out_dir / "logs" / f"build_{target_bpw:.2f}.log"
    rc = run_subprocess(cmd, log_path=log)
    if rc != 0:
        raise RuntimeError(f"quantize_mixed_budget.py target={target_bpw} exited with {rc}")
    return output


def run_ppl(
    target_bpw: float,
    checkpoint: Path,
    out_dir: Path,
    model_dir: Path,
    wikitext: Path,
    device: str,
) -> dict:
    """Run limited_ppl_bench.py on the given checkpoint. Returns parsed JSON."""
    result_path = out_dir / "ppl" / f"target_{target_bpw:.2f}.json"
    if result_path.exists():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        if "ppl" in data:
            print(f"[ppl {target_bpw}] result already exists: PPL={data['ppl']:.4f}")
            return data

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "limited_ppl_bench.py"),
        "--label", f"act_weighted_target_{target_bpw:.2f}",
        "--model-dir", str(model_dir),
        "--wikitext", str(wikitext),
        "--quantized-pt", str(checkpoint),
        "--tokens", str(PPL_TOKENS_CAP),
        "--max-length", str(PPL_MAX_LENGTH),
        "--stride", str(PPL_STRIDE),
        "--device", device,
        "--output", str(result_path),
    ]
    log = out_dir / "logs" / f"ppl_{target_bpw:.2f}.log"
    rc = run_subprocess(cmd, log_path=log)
    if rc != 0:
        raise RuntimeError(f"limited_ppl_bench.py target={target_bpw} exited with {rc}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--wikitext", default=str(DEFAULT_WIKITEXT))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--targets", default=",".join(f"{t}" for t in DEFAULT_TARGETS_BPW),
                   help="Comma-separated target BPW values, e.g. '3.75,3.5,3.0'")
    p.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    p.add_argument("--outlier-options", default=",".join(str(k) for k in DEFAULT_OUTLIER_OPTIONS))
    p.add_argument("--act-tokens", type=int, default=DEFAULT_ACT_TOKENS)
    p.add_argument("--act-max-length", type=int, default=DEFAULT_ACT_MAX_LENGTH)
    p.add_argument("--act-stride", type=int, default=DEFAULT_ACT_STRIDE)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-ppl", action="store_true",
                   help="Skip PPL stage (build and scan only).")
    p.add_argument("--ppl-tokens-cap", type=int, default=PPL_TOKENS_CAP)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global PPL_TOKENS_CAP
    PPL_TOKENS_CAP = args.ppl_tokens_cap

    model_dir = Path(args.model_dir)
    wikitext = Path(args.wikitext)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    targets = [float(t) for t in args.targets.split(",") if t.strip()]
    outlier_options = [int(t) for t in args.outlier_options.split(",") if t.strip()]

    print("=" * 72)
    print("ACTIVATION-WEIGHTED MIXED-BUDGET PARETO SCAN")
    print("=" * 72)
    print(f"Model dir:       {model_dir}")
    print(f"WikiText:        {wikitext}")
    print(f"Output dir:      {out_dir}")
    print(f"Targets (BPW):   {targets}")
    print(f"Group size:      {args.group_size}")
    print(f"Outlier options: {outlier_options}")
    print(f"Device:          {args.device}")
    print(f"Skip PPL:        {args.skip_ppl}")
    print()

    t0 = time.time()

    # Stage 1: collect activation weights
    act_path = ensure_activation_weights(
        out_dir=out_dir,
        model_dir=model_dir,
        wikitext=wikitext,
        act_tokens=args.act_tokens,
        max_length=args.act_max_length,
        stride=args.act_stride,
        device=args.device,
    )
    print(f"[act] activation weights: {act_path} (elapsed {time.time()-t0:.1f}s)")
    print()

    # Stage 2-4: per-target scan, build, ppl
    summary: dict = {
        "targets": targets,
        "group_size": args.group_size,
        "outlier_options": outlier_options,
        "device": args.device,
        "act_weights": str(act_path),
        "results": {},
    }

    for target_bpw in targets:
        print("=" * 72)
        print(f"TARGET BPW = {target_bpw:.2f}")
        print("=" * 72)
        t_target = time.time()

        scan_path = run_scan(
            target_bpw=target_bpw,
            out_dir=out_dir,
            model_dir=model_dir,
            activation_weights=act_path,
            group_size=args.group_size,
            outlier_options=outlier_options,
        )
        scan_data = json.loads(scan_path.read_text(encoding="utf-8"))
        alloc = scan_data.get("mixed_allocation", {})
        scan_summary = {
            "scan_path": str(scan_path),
            "avg_bpw": alloc.get("avg_bpw"),
            "weighted_rmse": alloc.get("weighted_rmse"),
            "method_counts": alloc.get("method_counts"),
            "layers": alloc.get("layers"),
        }
        print(f"[scan {target_bpw}] avg_bpw={scan_summary['avg_bpw']:.4f}, "
              f"weighted_rmse={scan_summary['weighted_rmse']:.6f}, "
              f"methods={scan_summary['method_counts']}")
        print(f"[scan {target_bpw}] elapsed {time.time()-t_target:.1f}s")

        checkpoint = run_build(
            target_bpw=target_bpw,
            scan_path=scan_path,
            out_dir=out_dir,
            model_dir=model_dir,
            group_size=args.group_size,
        )
        ckpt_size_mb = checkpoint.stat().st_size / 1e6
        print(f"[build {target_bpw}] {checkpoint} ({ckpt_size_mb:.1f} MB) "
              f"elapsed {time.time()-t_target:.1f}s")

        if args.skip_ppl:
            ppl_data = {"ppl": None, "skipped": True}
        else:
            ppl_data = run_ppl(
                target_bpw=target_bpw,
                checkpoint=checkpoint,
                out_dir=out_dir,
                model_dir=model_dir,
                wikitext=wikitext,
                device=args.device,
            )
            print(f"[ppl {target_bpw}] PPL = {ppl_data.get('ppl'):.4f} "
                  f"elapsed {time.time()-t_target:.1f}s")

        scan_summary.update({
            "checkpoint": str(checkpoint),
            "checkpoint_size_mb": ckpt_size_mb,
            "ppl": ppl_data.get("ppl"),
            "ppl_data": ppl_data if args.skip_ppl else {
                "ppl": ppl_data.get("ppl"),
                "seq_len": ppl_data.get("seq_len"),
                "chunks": ppl_data.get("chunks"),
            },
        })
        summary["results"][f"{target_bpw:.2f}"] = scan_summary
        print()

    # Stage 5: summary
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[summary] wrote {summary_path}")

    # Human-readable report
    report = []
    report.append("=" * 72)
    report.append("ACTIVATION-WEIGHTED MIXED-BUDGET PARETO RESULTS")
    report.append("=" * 72)
    bf16_ppl = 108.4542
    mixed_4p0_ppl = 107.5656
    report.append(f"BF16 baseline (Gemma-4-E2B / WikiText): PPL = {bf16_ppl:.4f}")
    report.append(f"Mixed-budget uniform 4.00 BPW (uniform allocator): PPL = {mixed_4p0_ppl:.4f}")
    report.append("")
    report.append(f"{'target':>8}  {'avg_bpw':>9}  {'size_mb':>8}  {'ppl':>10}  {'vs_bf16':>9}  {'vs_4p0':>9}")
    for k, r in summary["results"].items():
        ppl = r.get("ppl")
        ppl_str = f"{ppl:.4f}" if isinstance(ppl, (int, float)) else "n/a"
        delta_bf = (ppl - bf16_ppl) if isinstance(ppl, (int, float)) else None
        delta_4p0 = (ppl - mixed_4p0_ppl) if isinstance(ppl, (int, float)) else None
        dbf = f"{delta_bf:+.4f}" if delta_bf is not None else "n/a"
        d4p = f"{delta_4p0:+.4f}" if delta_4p0 is not None else "n/a"
        report.append(
            f"{k:>8}  {r['avg_bpw']:>9.4f}  {r['checkpoint_size_mb']:>8.1f}  "
            f"{ppl_str:>10}  {dbf:>9}  {d4p:>9}"
        )
    report.append("")
    report.append(f"Total elapsed: {time.time()-t0:.1f}s")
    report_text = "\n".join(report) + "\n"
    report_path = out_dir / "REPORT.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"[summary] wrote {report_path}")


if __name__ == "__main__":
    main()
