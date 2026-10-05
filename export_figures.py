#!/usr/bin/env python3
"""Build the editorial appendix from a checkpoint and its captured frames.

Example:
  python export_figures.py --checkpoint gpt_wikitext.pt --theme print,slides --width double

Reads the run history from the checkpoint, every frame captured for it (exports/frames-<name>/
by default, from the dashboard's Export tab or `gpt_lab.py --record-every N`), and the current
weights (measured once more on the validation passage), then writes exports/appendix-<name>-
step<N>-<time>/ with figures, strips, data sheets and cards (see export.py).
"""
from __future__ import annotations

import argparse
import os
import sys

import export as ex
from data import TokenData, prepare
from model import GPTConfig
from train import TrainConfig, Trainer, load_checkpoint, resolve_device


def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=os.path.join(here, "gpt_wikitext.pt"))
    p.add_argument("--frames", default=None, help="frames folder (default exports/frames-<checkpoint>)")
    p.add_argument("--out", default=None, help="appendix folder (default exports/appendix-...)")
    p.add_argument("--theme", default="print,slides")
    p.add_argument("--width", default="double", choices=("single", "onehalf", "double"))
    p.add_argument("--formats", default="png,pdf,svg")
    p.add_argument("--sections", default="ABCDE", help="any of A (cards) B (curves) C (maps) D (sheets) E (frames)")
    p.add_argument("--data-dir", default=os.path.join(here, "data"))
    p.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda", "mps"))
    args = p.parse_args(argv)
    try:
        ck = load_checkpoint(args.checkpoint)
        mcfg = GPTConfig(**ck["model_config"])
        tcfg = TrainConfig(**{k: v for k, v in ck["train_config"].items() if k in TrainConfig.__dataclass_fields__})
        data = TokenData(prepare(tcfg.dataset, args.data_dir, mcfg.vocab_size))
        tr = Trainer(mcfg, tcfg, data, device=resolve_device(args.device))
        tr.load_state(ck)
    except (ValueError, OSError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    tr.run_probes()
    frames = args.frames or ex.default_frames_dir(args.checkpoint)
    opts = ex.ExportOptions(themes=tuple(t for t in args.theme.split(",") if t in ex.THEMES), width=args.width,
                            formats=tuple(f for f in args.formats.split(",") if f in ("png", "pdf", "svg")),
                            sections=tuple(s for s in args.sections.upper() if s in ex.SECTIONS))
    out = args.out or ex.default_export_dir(args.checkpoint, tr.step)
    readme = ex.build_appendix(ex.context_from_trainer(tr, frames, args.checkpoint), out, opts,
                               progress=lambda m, f: print(f"{100 * f:3.0f}% {m}", flush=True))
    print(f"appendix: {readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
