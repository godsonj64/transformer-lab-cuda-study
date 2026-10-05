#!/usr/bin/env python3
"""
gpt_lab.py - train a GPT-style transformer language model on WikiText-103 with a live
dashboard, or headless. Training can be paused, saved, resumed and continued.

    python gpt_lab.py                          # dashboard (resumes gpt_wikitext.pt if it exists)
    python gpt_lab.py --headless --max-minutes 60
    python gpt_lab.py --resume gpt_wikitext.pt --max-steps 40000     # continue further
    python gpt_lab.py --fresh --save other.pt  # a new run next to an existing one
    python gpt_lab.py --prepare-only           # download + tokenize the corpus, then exit
    python gpt_lab.py --help

The first run downloads WikiText-103 (raw) from Hugging Face (about 315 MB, SHA-256
checked), learns a 16,384-token byte-level BPE on the training split and tokenizes the
corpus into data/wikitext-103/ (about 250 MB). That takes about a minute on an Apple M4.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

from data import TokenData, prepare
from model import GPTConfig
from train import (RunLogger, TrainConfig, Trainer, default_run_dir, load_checkpoint, merge_train_config,
                   resolve_device, run_headless, summary_line)

HERE = os.path.dirname(os.path.abspath(__file__))       # defaults live next to this script, wherever it runs from
DEFAULT_SAVE = os.path.join(HERE, "gpt_wikitext.pt")
DEFAULT_DATA = os.path.join(HERE, "data")
DEFAULT_RUNS = os.path.join(HERE, "runs")
MODEL_FLAGS = {"n_layer": "n_layer", "n_head": "n_head", "d_model": "d_model", "block_size": "block_size",
               "mlp": "mlp", "norm": "norm", "pos": "pos", "dropout": "dropout", "mlp_hidden": "mlp_hidden"}
TRAIN_FLAGS = ("batch_size", "grad_accum", "max_steps", "lr", "min_lr_ratio", "warmup_steps", "schedule",
               "decay_frac", "weight_decay", "grad_clip", "eval_interval", "eval_tokens", "probe_interval",
               "sample_interval", "seed", "device", "dtype")


def _positive_int(v: str) -> int:
    i = int(v)
    if i < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return i


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    m, t = GPTConfig(), TrainConfig()
    p = argparse.ArgumentParser(description="GPT-style transformer on WikiText-103 with a live dashboard.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--dataset", default=None,
                   help=f"wikitext-103, wikitext-2 or a UTF-8 .txt file (default {t.dataset}; a resumed run keeps its own)")
    g.add_argument("--data-dir", default=DEFAULT_DATA)
    g.add_argument("--vocab-size", type=_positive_int, default=None, help=f"BPE vocabulary (default {m.vocab_size})")
    g.add_argument("--workers", type=_positive_int, default=None, help="processes for tokenizer training")
    g.add_argument("--prepare-only", action="store_true", help="prepare the corpus and exit")
    g = p.add_argument_group("model (fresh runs; a resumed run keeps its architecture)")
    g.add_argument("--n-layer", type=_positive_int, default=None, help=f"blocks (default {m.n_layer})")
    g.add_argument("--n-head", type=_positive_int, default=None, help=f"attention heads (default {m.n_head})")
    g.add_argument("--d-model", type=_positive_int, default=None, help=f"residual width (default {m.d_model})")
    g.add_argument("--block-size", type=_positive_int, default=None, help=f"context length (default {m.block_size})")
    g.add_argument("--mlp", choices=("swiglu", "gelu"), default=None, help=f"default {m.mlp}")
    g.add_argument("--mlp-hidden", type=int, default=None, help="MLP width (0 = 8/3 d for SwiGLU, 4 d for GELU)")
    g.add_argument("--norm", choices=("rms", "layer"), default=None, help=f"default {m.norm}")
    g.add_argument("--pos", choices=("rope", "learned"), default=None, help=f"default {m.pos}")
    g.add_argument("--dropout", type=float, default=None, help=f"default {m.dropout}")
    g = p.add_argument_group("training (explicit flags override a resumed run's settings)")
    g.add_argument("--batch-size", type=_positive_int, default=None, help=f"sequences per micro-batch ({t.batch_size})")
    g.add_argument("--grad-accum", type=_positive_int, default=None, help=f"micro-batches per update ({t.grad_accum})")
    g.add_argument("--max-steps", type=_positive_int, default=None, help=f"schedule length and stop ({t.max_steps})")
    g.add_argument("--lr", type=float, default=None, help=f"peak learning rate ({t.lr})")
    g.add_argument("--min-lr-ratio", type=float, default=None, help=f"final / peak lr ({t.min_lr_ratio})")
    g.add_argument("--warmup-steps", type=int, default=None, help=f"linear warm-up ({t.warmup_steps})")
    g.add_argument("--schedule", choices=("cosine", "wsd", "constant"), default=None,
                   help=f"({t.schedule}); wsd = warm-up, stable, decay: suits open-ended runs")
    g.add_argument("--decay-frac", type=float, default=None, help=f"wsd decay fraction ({t.decay_frac})")
    g.add_argument("--weight-decay", type=float, default=None, help=f"AdamW decoupled decay ({t.weight_decay})")
    g.add_argument("--grad-clip", type=float, default=None, help=f"global norm clip, 0 = off ({t.grad_clip})")
    g.add_argument("--seed", type=int, default=None, help=f"({t.seed})")
    g.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None, help=f"({t.device})")
    g.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default=None,
                   help="autocast precision (auto: bf16 on CUDA, fp32 elsewhere)")
    g = p.add_argument_group("measurements")
    g.add_argument("--eval-interval", type=int, default=None, help=f"steps between validation passes ({t.eval_interval})")
    g.add_argument("--eval-tokens", type=int, default=None, help="validation tokens per pass (0 = all)")
    g.add_argument("--probe-interval", type=int, default=None, help=f"steps between probe snapshots ({t.probe_interval})")
    g.add_argument("--sample-interval", type=int, default=None, help=f"steps between samples ({t.sample_interval})")
    g = p.add_argument_group("checkpoints and logs")
    g.add_argument("--save", default=DEFAULT_SAVE, metavar="PATH", help="checkpoint path (S key, autosave, exit)")
    g.add_argument("--resume", metavar="PATH", default=None, help="continue from this checkpoint")
    g.add_argument("--fresh", action="store_true", help="start a new run even if --save exists")
    g.add_argument("--overwrite", action="store_true", help="allow --fresh to replace an existing --save file")
    g.add_argument("--autosave-minutes", type=float, default=10.0, help="0 = only on S and on exit")
    g.add_argument("--no-save", action="store_true", help="never write checkpoints")
    g.add_argument("--log-dir", default=None, help="logs folder (default runs/<timestamp> next to this script)")
    g.add_argument("--no-log", action="store_true")
    g = p.add_argument_group("headless")
    g.add_argument("--headless", action="store_true", help="train without a window")
    g.add_argument("--max-minutes", type=float, default=None, help="wall-clock budget")
    g.add_argument("--max-tokens", type=float, default=None, help="stop after this many training tokens")
    g.add_argument("--status-every", type=float, default=15.0, help="seconds between status lines")
    g = p.add_argument_group("time-lapse capture and appendix export (see export.py)")
    g.add_argument("--record-every", type=int, default=0, help="capture a frame every N updates (0 = off)")
    g.add_argument("--record-from", type=int, default=None, help="first update to capture (default: the start)")
    g.add_argument("--record-until", type=int, default=None, help="last update to capture")
    g.add_argument("--record-max", type=int, default=40, help="most frames to capture in this session")
    g.add_argument("--record-panels", action="store_true",
                   help="headless: also render and save every dashboard panel (offscreen) at each frame")
    g.add_argument("--frames-dir", default=None, help="frames folder (default exports/frames-<checkpoint>)")
    g.add_argument("--export-on-exit", action="store_true", help="build the appendix when the run ends")
    g.add_argument("--export-theme", default="print,slides", help="print, slides or print,slides")
    g.add_argument("--export-width", default="double", choices=("single", "onehalf", "double"))
    g.add_argument("--export-formats", default="png,pdf,svg")
    g.add_argument("--export-dir", default=None, help="appendix folder (default exports/appendix-...)")
    g = p.add_argument_group("dashboard")
    g.add_argument("--fps", type=_positive_int, default=30)
    g.add_argument("--view", default="training", choices=("training", "data", "predict", "attention", "heads",
                                                           "lens", "network", "embed", "generate"))
    g.add_argument("--paused", action="store_true", help="open the dashboard with training paused")
    return p.parse_args(argv)


def build(args: argparse.Namespace) -> tuple[Trainer, str | None, list[str]]:
    """The trainer for this invocation (fresh or resumed) and notes to print."""
    notes: list[str] = []
    resume = args.resume
    if resume is None and not args.fresh and args.save and os.path.exists(args.save):
        resume = args.save
        notes.append(f"resuming {args.save} (use --fresh --save NEW.pt for a new run)")
    if resume is None and args.fresh and args.save and os.path.exists(args.save) and not (args.overwrite or args.no_save):
        raise ValueError(f"{args.save} exists; pass --overwrite to replace it or --save another path")
    overrides = {k: getattr(args, k) for k in TRAIN_FLAGS}
    if resume is not None:
        ck = load_checkpoint(resume)
        mcfg = GPTConfig(**ck["model_config"])
        ignored = [f"--{k.replace('_', '-')}" for k in MODEL_FLAGS if getattr(args, k) is not None
                   and getattr(args, k) != getattr(mcfg, k)]
        if ignored:
            notes.append(f"ignored for a resumed run (architecture is fixed): {', '.join(ignored)}")
        tcfg, changed = merge_train_config(ck["train_config"], overrides)
        if args.dataset and args.dataset != tcfg.dataset:
            raise ValueError(f"checkpoint was trained on {tcfg.dataset}, not {args.dataset}")
        if changed:
            notes.append("changed on resume: " + "; ".join(changed))
            if any(c.startswith(("batch_size", "grad_accum", "seed")) for c in changed):
                notes.append("batch order now differs from an uninterrupted run (batch size or seed changed)")
            if any(c.startswith(("max_steps", "schedule", "warmup_steps", "lr", "min_lr_ratio")) for c in changed):
                notes.append("the learning-rate schedule changed: the curve may jump at the resume step")
        info = prepare(tcfg.dataset, args.data_dir, mcfg.vocab_size, args.workers)
        data = TokenData(info)
        tr = Trainer(mcfg, tcfg, data, device=resolve_device(tcfg.device))
        tr.load_state(ck)
        notes.append(f"resumed at step {tr.step:,} ({tr.tokens / 1e6:.1f}M tokens)")
    else:
        mdict = {k: getattr(args, k) for k in MODEL_FLAGS if getattr(args, k) is not None}
        vocab = args.vocab_size or GPTConfig().vocab_size
        tvals = {k: v for k, v in overrides.items() if v is not None}
        tcfg = TrainConfig(dataset=args.dataset or TrainConfig().dataset, data_dir=args.data_dir, **tvals)
        info = prepare(tcfg.dataset, args.data_dir, vocab, args.workers)
        data = TokenData(info)
        mcfg = GPTConfig(vocab_size=vocab, **mdict)
        tr = Trainer(mcfg, tcfg, data, device=resolve_device(tcfg.device))
    return tr, resume, notes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.prepare_only:
            info = prepare(args.dataset or TrainConfig().dataset, args.data_dir,
                           args.vocab_size or GPTConfig().vocab_size, args.workers)
            print(f"ready: {info.dir}")
            return 0
        tr, resume, notes = build(args)
    except (ValueError, OSError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for n in notes:
        print(n, flush=True)
    m = tr.model
    print(f"model: {tr.model_cfg.n_layer} layers, {tr.model_cfg.n_head} heads, d={tr.model_cfg.d_model}, "
          f"context {tr.model_cfg.block_size}, {m.num_params() / 1e6:.2f}M parameters "
          f"({m.num_params(True) / 1e6:.2f}M non-embedding) on {tr.device.type}", flush=True)
    if not args.no_log:
        config = {"model": tr.model_cfg.to_dict(), "train": asdict(tr.cfg), "data": {
            "dataset": tr.data.name, "fingerprint": tr.data.fingerprint, "splits": tr.data.meta.get("splits"),
            "baselines": tr.data.meta.get("baselines")}, "resumed_from": resume, "start_step": tr.step,
            "argv": sys.argv[1:] if argv is None else argv}
        tr.logger = RunLogger(args.log_dir or default_run_dir(DEFAULT_RUNS), config)
        print(f"logs: {tr.logger.dir}", flush=True)
    save_path = None if args.no_save else args.save
    import export as ex
    frames_dir = args.frames_dir or ex.default_frames_dir(save_path or args.save)
    try:
        if args.headless:
            secs = args.max_minutes * 60 if args.max_minutes is not None else None
            on_step = None
            if args.record_every > 0:
                rec = ex.FrameRecorder(frames_dir, every=args.record_every,
                                       from_step=args.record_from if args.record_from is not None else tr.step,
                                       until_step=args.record_until, panels=args.record_panels)
                rec.max_frames = len(rec.captured) + args.record_max
                painter = offscreen_painter(tr, frames_dir, rec) if args.record_panels else None

                def on_step(t):
                    if rec.due(t.step):
                        if t.snapshot is None or t.snapshot.get("step") != t.step:
                            t.run_probes()
                        rec.capture_data(t)
                        if painter is not None:
                            painter(t.step)
                        print(f"captured frame {len(rec.captured)} at step {t.step:,} -> {rec.step_dir(t.step)}",
                              flush=True)
                print(f"recording every {args.record_every} updates into {frames_dir}", flush=True)
            reason = run_headless(tr, save_path, secs, int(args.max_tokens) if args.max_tokens else None,
                                  args.status_every, args.autosave_minutes, on_step)
            print(f"stopped ({reason}): {summary_line(tr)}", flush=True)
        else:
            from dashboard import LabGUI, Worker
            worker = Worker(tr, save_path, args.autosave_minutes, threaded=True, start_paused=args.paused)
            worker.frames_dir = frames_dir
            gui = LabGUI(worker, save_path or DEFAULT_SAVE, fps=args.fps, view=args.view, autosave=save_path is not None)
            if args.record_every > 0:
                gui.rec_cfg.update(every=args.record_every, until=args.record_until, max_frames=args.record_max)
                worker.submit("record", on=True, every=args.record_every, until_step=args.record_until,
                              max_frames=args.record_max, panels=True,
                              from_step=args.record_from if args.record_from is not None else tr.step)
            gui.run()
            print(summary_line(tr), flush=True)
        if args.export_on_exit:
            if tr.snapshot is None:
                tr.run_probes()
            opts = ex.ExportOptions(themes=tuple(t for t in args.export_theme.split(",") if t in ex.THEMES),
                                    width=args.export_width,
                                    formats=tuple(f for f in args.export_formats.split(",") if f in ("png", "pdf", "svg")))
            out = args.export_dir or ex.default_export_dir(save_path or args.save, tr.step)
            readme = ex.build_appendix(ex.context_from_trainer(tr, frames_dir, save_path), out, opts,
                                       progress=lambda m, f: print(f"  export {100 * f:3.0f}% {m}", flush=True))
            print(f"appendix: {readme}", flush=True)
    finally:
        tr.close()
    return 0


def offscreen_painter(tr, frames_dir: str, rec):
    """Render the dashboard without a window (SDL dummy driver) to save panel frames."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from dashboard import LabGUI, Worker
    worker = Worker(tr, None, 0, threaded=False)
    worker.frames_dir = frames_dir
    worker.recorder = rec
    gui = LabGUI(worker, os.path.join(frames_dir, "unused.pt"), fps=1000, autosave=False)
    return gui.capture_dashboard


if __name__ == "__main__":
    sys.exit(main())
