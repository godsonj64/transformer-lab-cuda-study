"""Cross-platform runner for the paper study (Windows / Linux / macOS).

Runs the RoPE vs learned-absolute-position study with exactly the flags used on the
Mac (paper_study/run_queue.sh), then evaluates each finished checkpoint once on the
validation and test splits. Finished runs are marked with paper_study/done_<name> and
skipped on the next start, so the runner can be stopped (Ctrl+C saves) and restarted.

    python paper_study/run_study.py --device cuda          (seeds 0 1 2)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOKENS_PER_UPDATE = 16 * 512


def run(cmd: list[str], log_path: str) -> int:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    print(">", " ".join(cmd), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return p.wait()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--conditions", nargs="+", default=["rope", "learned"], choices=["rope", "learned"])
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="float32 matches the Mac runs; keep it for the paper")
    ap.add_argument("--no-eval", action="store_true", help="skip the final validation/test evaluation")
    a = ap.parse_args()

    if a.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("CUDA is not available to this PyTorch. Run setup_windows.bat first (see README_WINDOWS.md).")
            return 2
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        arch = f"sm_{cap[0]}{cap[1]}"
        print(f"GPU: {name} ({arch}); torch {torch.__version__}, CUDA {torch.version.cuda}")
        if arch not in torch.cuda.get_arch_list():
            print(f"This PyTorch build has no kernels for {arch} ({torch.cuda.get_arch_list()}). "
                  "RTX 50-series GPUs need a CUDA 12.8+ build; rerun setup_windows.bat.")
            return 2

    py = sys.executable
    for d in ("ckpt", "logs", "frames", "appendix", "eval"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    queue_log = os.path.join(HERE, f"queue_{a.device}.log")
    for seed in a.seeds:
        for cond in a.conditions:
            name = f"{cond}_s{seed}"
            ck = os.path.join("paper_study", "ckpt", f"{name}.pt")
            done = os.path.join(HERE, f"done_{name}")
            if not os.path.exists(done):
                resume = os.path.exists(os.path.join(ROOT, ck))
                print(f"=== START {name} {time.ctime()} ({'resuming' if resume else 'fresh'})", flush=True)
                cmd = [py, "gpt_lab.py", "--headless", "--device", a.device, "--dtype", a.dtype,
                       "--pos", cond, "--seed", str(seed), "--max-tokens", str(a.updates * TOKENS_PER_UPDATE),
                       "--eval-interval", "100", "--probe-interval", "25", "--sample-interval", "1000",
                       "--record-every", "500", "--record-max", "10", "--export-on-exit",
                       "--export-theme", "print", "--export-formats", "png,pdf",
                       "--export-dir", os.path.join("paper_study", "appendix", name),
                       "--save", ck, "--log-dir", os.path.join("paper_study", "logs", name),
                       "--frames-dir", os.path.join("paper_study", "frames", name)]
                if not resume:
                    cmd[2:2] = ["--fresh", "--overwrite"]
                rc = run(cmd, queue_log)
                print(f"=== END {name} {time.ctime()} rc={rc}", flush=True)
                if rc != 0:
                    print("Stopped. Run the same command again to resume this run.")
                    return rc
                with open(done, "w", encoding="utf-8") as f:
                    f.write(f"{time.ctime()} device={a.device} dtype={a.dtype}\n")
            out = os.path.join("paper_study", "eval", name)
            if not a.no_eval and not os.path.exists(os.path.join(ROOT, out, "summary.json")):
                rc = run([py, "evaluate_checkpoint.py", "--checkpoint", ck, "--split", "validation,test",
                          "--stride", "256", "--device", a.device, "--out-dir", out], queue_log)
                if rc != 0:
                    return rc
    print(f"=== ALL DONE {time.ctime()}. Next: python paper_study/pack_results.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
