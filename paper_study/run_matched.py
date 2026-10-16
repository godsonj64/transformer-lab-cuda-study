"""Matched-initialization runs to 6,000 updates for the paper study (Windows / Linux / macOS).

Learned-position runs are retrained from scratch with --pos-init last: the position table
is drawn after every other weight, so at each seed every other initial weight is
bit-identical to the RoPE run's. The RoPE model's initialization is unchanged by this, so
RoPE runs are continued from the original 3,000-update checkpoints in paper_study/ckpt when
those still hold the optimizer state, and otherwise retrained from scratch (same initial
weights and data order as the original run).

All settings except the length are those of run_study.py. Everything is written to
paper_study/matched6000/, so the original runs are not touched. Finished runs are marked
with matched6000/done_<name> and skipped on the next start, so the runner can be stopped
(Ctrl+C saves) and restarted. At the end each checkpoint is evaluated once on the
validation and test splits, and slim checkpoints, logs, evaluations and figures are packed
into paper_study/matched6000_<host>_<time>.zip for the analysis machine.

    python paper_study/run_matched.py --device cuda          (seeds 0 1 2, learned first)
"""
from __future__ import annotations

import argparse
import glob
import os
import platform
import sys
import time
import zipfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from run_study import TOKENS_PER_UPDATE, check_cuda, run  # noqa: E402

OUT = "matched6000"


def has_optimizer(path: str) -> bool:
    if not os.path.exists(path):
        return False
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return "optimizer" in ck and not ck.get("slim", False)


def pack(names: list[str]) -> str:
    base = os.path.join(HERE, OUT)
    out = os.path.join(HERE, f"{OUT}_{platform.node() or 'host'}_{time.strftime('%Y%m%d-%H%M')}.zip")
    slim = os.path.join(base, "_slim.pt")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in names:
            ck = torch.load(os.path.join(base, "ckpt", f"{name}.pt"), map_location="cpu", weights_only=False)
            ck.pop("optimizer", None)
            ck["slim"] = True
            torch.save(ck, slim)
            z.write(slim, f"paper_study/{OUT}/ckpt/{name}.pt")
            for p in [os.path.join(base, f"done_{name}")] + glob.glob(os.path.join(base, "*", name, "**", "*"), recursive=True):
                if os.path.isfile(p):
                    z.write(p, os.path.relpath(p, ROOT).replace(os.sep, "/"))
            print(f"packed {name}")
        for p in glob.glob(os.path.join(base, "queue_*.log")):
            z.write(p, os.path.relpath(p, ROOT).replace(os.sep, "/"))
    os.remove(slim)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--conditions", nargs="+", default=["learned", "rope"], choices=["rope", "learned"])
    ap.add_argument("--updates", type=int, default=6000)
    ap.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="float32 matches the original runs; keep it for the paper")
    a = ap.parse_args()
    if a.device == "cuda" and not check_cuda():
        return 2

    py = sys.executable
    base = os.path.join("paper_study", OUT)
    for d in ("ckpt", "logs", "frames", "appendix", "eval"):
        os.makedirs(os.path.join(ROOT, base, d), exist_ok=True)
    queue_log = os.path.join(ROOT, base, f"queue_{a.device}.log")
    names = [f"{c}_s{s}" for c in a.conditions for s in a.seeds]
    for name in names:
        cond, seed = name.split("_s")
        ck = os.path.join(base, "ckpt", f"{name}.pt")
        done = os.path.join(ROOT, base, f"done_{name}")
        if not os.path.exists(done):
            original = os.path.join("paper_study", "ckpt", f"{name}.pt")
            if os.path.exists(os.path.join(ROOT, ck)):
                src, how = ck, "resuming"
            elif cond == "rope" and has_optimizer(os.path.join(ROOT, original)):
                src, how = original, f"continuing {original}"
            else:
                src, how = None, "fresh"
            print(f"=== START {name} {time.ctime()} ({how})", flush=True)
            cmd = [py, "gpt_lab.py", "--headless", "--device", a.device, "--dtype", a.dtype,
                   "--max-tokens", str(a.updates * TOKENS_PER_UPDATE),
                   "--eval-interval", "100", "--probe-interval", "25", "--sample-interval", "1000",
                   "--record-every", "500", "--record-max", str(a.updates // 500), "--export-on-exit",
                   "--export-theme", "print", "--export-formats", "png,pdf",
                   "--export-dir", os.path.join(base, "appendix", name),
                   "--save", ck, "--log-dir", os.path.join(base, "logs", name),
                   "--frames-dir", os.path.join(base, "frames", name)]
            if src is None:
                cmd[2:2] = ["--fresh", "--overwrite", "--pos", cond, "--pos-init", "last", "--seed", seed]
            else:
                cmd[2:2] = ["--resume", src]
            rc = run(cmd, queue_log)
            print(f"=== END {name} {time.ctime()} rc={rc}", flush=True)
            if rc != 0:
                print("Stopped. Run the same command again to resume this run.")
                return rc
            with open(done, "w", encoding="utf-8") as f:
                f.write(f"{time.ctime()} device={a.device} dtype={a.dtype} start={how}\n")
        out = os.path.join(base, "eval", name)
        if not os.path.exists(os.path.join(ROOT, out, "summary.json")):
            rc = run([py, "evaluate_checkpoint.py", "--checkpoint", ck, "--split", "validation,test",
                      "--stride", "256", "--device", a.device, "--out-dir", out], queue_log)
            if rc != 0:
                return rc
    zp = pack(names)
    print(f"=== ALL DONE {time.ctime()}. Copy {zp} ({os.path.getsize(zp) / 1e6:.0f} MB) to the analysis machine.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
