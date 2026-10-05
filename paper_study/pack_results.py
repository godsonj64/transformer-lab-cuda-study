"""Pack finished study runs into one zip to copy back to the Mac.

Checkpoints are slimmed: the AdamW moment buffers are dropped (weights, configuration,
counters and the full metric history are kept), which cuts each file from ~195 MB to
~70 MB. The slim files are enough for analyze.py and final_probes.py; they cannot be
used to continue training.

    python paper_study/pack_results.py            -> paper_study/results_<host>_<time>.zip
"""
from __future__ import annotations

import glob
import os
import platform
import sys
import time
import zipfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    done = sorted(os.path.basename(p)[5:] for p in glob.glob(os.path.join(HERE, "done_*")))
    if not done:
        print("no finished runs (paper_study/done_*)")
        return 1
    out = os.path.join(HERE, f"results_{platform.node() or 'host'}_{time.strftime('%Y%m%d-%H%M')}.zip")
    slim_dir = os.path.join(HERE, "_slim")
    os.makedirs(slim_dir, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in done:
            ck_path = os.path.join(HERE, "ckpt", f"{name}.pt")
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            ck.pop("optimizer", None)
            ck["slim"] = True
            slim = os.path.join(slim_dir, f"{name}.pt")
            torch.save(ck, slim)
            z.write(slim, f"paper_study/ckpt/{name}.pt")
            z.write(os.path.join(HERE, f"done_{name}"), f"paper_study/done_{name}")
            for sub in ("logs", "eval", "appendix", "frames"):
                base = os.path.join(HERE, sub, name)
                for p in glob.glob(os.path.join(base, "**", "*"), recursive=True):
                    if os.path.isfile(p):
                        z.write(p, os.path.relpath(p, os.path.dirname(HERE)).replace(os.sep, "/"))
            print(f"packed {name}")
        for p in glob.glob(os.path.join(HERE, "queue_*.log")):
            z.write(p, f"paper_study/{os.path.basename(p)}")
    for p in glob.glob(os.path.join(slim_dir, "*.pt")):
        os.remove(p)
    os.rmdir(slim_dir)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB). Copy it to the Mac for analysis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
