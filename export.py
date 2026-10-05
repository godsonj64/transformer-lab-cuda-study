"""
export.py - editorial exports for manuscripts and presentations.

Capture (time-lapse)
    FrameRecorder saves, every N updates inside a chosen step window, one "frame" per capture
    step: the measurements at that step (attention, logit lens, head scores, embedding PCA,
    block internals, the latest evaluation) as compressed arrays, and optionally a PNG of
    every dashboard panel. Frames accumulate in one folder per checkpoint, across sessions.

Appendix (export)
    build_appendix() turns the run history and the captured frames into a folder laid out
    like a manuscript appendix:

        README.md              index: every figure and table with its legend and files
        manifest.json          provenance: checkpoint, step, tokens, data fingerprint, frames
        A_data_cards/          run card (key numbers), model card, data card
        B_curves/              training dynamics; related curves combined as lettered strips
        C_maps/                attention, head scores, logit lens, layer health, embeddings,
                               block internals; time-lapse strips across captured steps
        D_data_sheets/         tables as CSV + Markdown + typeset table figures
        E_dashboard_frames/    per-panel time-lapse strips of the captured dashboard frames

    Figures come in two themes: "print" (white, for manuscripts; 89 mm single or 183 mm
    double column; 300-dpi PNG, PDF and SVG with embedded / editable text) and "slides"
    (black, matching the dashboard; 16:9 at 1920 x 1080). Panels in a strip carry letters
    a, b, c ...; time-lapse strips share one colour scale so frames are comparable. Every
    legend is generated from the measured numbers.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

FRAME_FORMAT = "transformer_lab_frame_v1"
WIDTHS_MM = {"single": 89.0, "onehalf": 120.0, "double": 183.0}
SLIDE_IN = (1920 / 144, 1080 / 144)              # 16:9: exactly 1920 x 1080 at 144 dpi
SECTIONS = {"A": "A_data_cards", "B": "B_curves", "C": "C_maps", "D": "D_data_sheets", "E": "E_dashboard_frames"}
ATT_N = 48                                       # tokens kept from attention maps in a frame
LENS_N = 32                                      # positions kept from the logit lens
HEAT_ANCHORS = [(40, 60, 190), (40, 140, 240), (70, 210, 230), (250, 225, 90), (250, 120, 40), (225, 30, 35)]

THEMES = {
    "print": dict(bg="#ffffff", fg="#111111", dim="#4d4d4d", faint="#7a7a7a", grid="#e4e4e4", spine="#333333",
                  train="#1a1a1a", raw="#b8b8b8", val="#0072B2", ref="#7a7a7a", lr="#E69F00", top1="#0072B2",
                  top5="#E69F00", accent="#D55E00", good="#009E73", card="#f4f5f6",
                  base=7.0, title=7.5, letter=9.0, line=0.9, thin=0.5, dpi=300),
    "slides": dict(bg="#000000", fg="#f7f9fa", dim="#c8ced1", faint="#a4aaad", grid="#2a2e30", spine="#9aa0a3",
                   train="#f7f9fa", raw="#5e6467", val="#48d8ff", ref="#a4aaad", lr="#fcd640", top1="#48d8ff",
                   top5="#fcd640", accent="#ee6854", good="#58de80", card="#101214",
                   base=14.0, title=16.0, letter=20.0, line=2.0, thin=1.0, dpi=144),
}


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
def frame_data(trainer) -> dict:
    """The measurements of one capture step, compacted for storage (float16 maps)."""
    snap, probe, emb = trainer.snapshot, trainer.last_probe, trainer.embed
    out = {"format": FRAME_FORMAT, "step": trainer.step, "tokens": trainer.tokens}
    if snap is not None:
        n = min(ATT_N, snap["attn"].shape[-1])
        m = min(LENS_N, len(snap["nll"]))
        out.update({
            "snap_step": snap["step"], "tokens_ids": snap["tokens"][: max(n, m) + 1].astype(np.int32),
            "nll": snap["nll"].astype(np.float32), "attn": snap["attn"][:, :, :n, :n].astype(np.float16),
            "head_entropy": snap["head_entropy"], "attn_by_distance": snap["attn_by_distance"],
            "lens_top": snap["lens_top"][:, :m].astype(np.int32), "lens_p_true": snap["lens_p_true"][:, :m],
            "lens_nll_mean": snap["lens_nll"].mean(axis=1), "resid_rms_mean": snap["resid_rms"].mean(axis=1),
            "attn_out_rms": snap["attn_out_rms"].mean(axis=1), "mlp_out_rms": snap["mlp_out_rms"].mean(axis=1),
            "top_id": snap["top_id"][:m, :3].astype(np.int32), "top_p": snap["top_p"][:m, :3],
        })
        for k, v in snap.get("block", {}).items():
            out[f"block_{k}"] = v.astype(np.float16)
        out["resid"] = snap["resid"][:, :32].astype(np.float16)
    if probe is not None:
        out.update({"induction": probe["induction"], "previous": probe["previous"], "duplicate": probe["duplicate"],
                    "loss_first": probe["loss_first"], "loss_repeat": probe["loss_repeat"],
                    "probe_seq_len": probe["seq_len"]})
        il, ih = np.unravel_index(int(np.argmax(probe["induction"])), probe["induction"].shape)
        out["probe_example"] = probe["example_attn"][il, ih].astype(np.float16)
        out["probe_example_head"] = np.array([il, ih])
    if emb is not None:
        out.update({"emb_ids": emb["ids"].astype(np.int32), "emb_coords": emb["coords"].astype(np.float16),
                    "emb_explained": emb["explained"], "emb_norms": emb["norms"].astype(np.float16),
                    "emb_counts": np.asarray(emb.get("counts", np.zeros(len(emb["ids"]))), dtype=np.float64)})
    e = trainer.last_eval
    if e is not None:
        out["eval_json"] = json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                       for k, v in e.items() if not isinstance(v, (np.ndarray, list))})
        if isinstance(e.get("pos_loss"), (np.ndarray, list)):
            out["eval_pos_loss"] = np.asarray(e["pos_loss"], dtype=np.float32)
    return out


@dataclass
class FrameRecorder:
    """Captures a frame every `every` updates while from_step <= step <= until_step."""
    directory: str
    every: int = 250
    from_step: int = 0
    until_step: int | None = None
    panels: bool = True
    max_frames: int = 40
    active: bool = True
    captured: list[int] = field(default_factory=list)
    last_message: str = ""

    def __post_init__(self):
        os.makedirs(self.directory, exist_ok=True)
        self.captured = sorted(self.existing_steps())

    def existing_steps(self) -> list[int]:
        out = []
        for name in os.listdir(self.directory):
            if name.startswith("step_") and os.path.exists(os.path.join(self.directory, name, "data.npz")):
                try:
                    out.append(int(name[5:]))
                except ValueError:
                    pass
        return out

    def step_dir(self, step: int) -> str:
        return os.path.join(self.directory, f"step_{step:07d}")

    def due(self, step: int) -> bool:
        if not self.active or step in self.captured or step < self.from_step:
            return False
        if self.until_step is not None and step > self.until_step:
            return False
        if len(self.captured) >= self.max_frames:
            self.active = False
            self.last_message = f"recording stopped: reached {self.max_frames} frames"
            return False
        return (step - self.from_step) % max(1, self.every) == 0

    MIN_FREE = 500e6                             # stop recording rather than fill the disk

    def capture_data(self, trainer) -> str:
        free = shutil.disk_usage(self.directory).free
        if free < self.MIN_FREE:
            self.active = False
            self.last_message = f"recording stopped: only {free / 1e6:.0f} MB free on the disk"
            raise OSError(self.last_message)
        d = self.step_dir(trainer.step)
        os.makedirs(d, exist_ok=True)
        data = frame_data(trainer)
        meta = {"format": FRAME_FORMAT, "step": trainer.step, "tokens": trainer.tokens,
                "captured_at": datetime.now().isoformat(timespec="seconds"), "dataset": trainer.data.name,
                "data_fingerprint": trainer.data.fingerprint}
        np.savez_compressed(os.path.join(d, "data.npz"), **{k: np.asarray(v) for k, v in data.items()})
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if trainer.step not in self.captured:
            self.captured.append(trainer.step)
            self.captured.sort()
        self.last_message = f"captured frame at step {trainer.step:,}"
        return d

    def disk_bytes(self) -> int:
        total = 0
        for root, _, files in os.walk(self.directory):
            total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        return total


def load_frames(directory: str | None) -> list[dict]:
    frames = []
    if not directory or not os.path.isdir(directory):
        return frames
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name, "data.npz")
        if name.startswith("step_") and os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                fr = {k: z[k] for k in z.files}
            fr["step"] = int(fr["step"])
            fr["tokens"] = int(fr["tokens"])
            fr["panels_dir"] = os.path.join(directory, name, "panels")
            if "eval_json" in fr:
                fr["eval"] = json.loads(str(fr["eval_json"]))
            frames.append(fr)
    frames.sort(key=lambda f: f["step"])
    return frames


# --------------------------------------------------------------------------- #
# Context: everything a figure needs, copied from the trainer
# --------------------------------------------------------------------------- #
@dataclass
class ExportContext:
    hist: dict
    meta: dict
    model_cfg: dict
    train_cfg: dict
    params: list[tuple[str, tuple, int]]
    n_params: int
    n_params_non_emb: int
    flops_per_token: float
    step: int
    tokens: int
    train_seconds: float
    tokens_per_epoch: int
    dataset: str
    fingerprint: str
    tokenizer: object
    frames: list[dict]
    checkpoint: str | None = None


def context_from_trainer(trainer, frames_dir: str | None = None, checkpoint: str | None = None) -> ExportContext:
    """Snapshot what the exports need. The current state is appended as a final frame when
    it is newer than the last captured one, so maps always include the latest step."""
    hist = {k: list(v) for k, v in trainer.hist.items()}
    frames = load_frames(frames_dir)
    if trainer.snapshot is not None and (not frames or frames[-1]["step"] < trainer.step):
        fr = {k: (np.asarray(v) if not isinstance(v, str) else v) for k, v in frame_data(trainer).items()}
        fr["step"], fr["tokens"] = int(trainer.step), int(trainer.tokens)
        fr["panels_dir"] = None
        if "eval_json" in fr:
            fr["eval"] = json.loads(str(fr["eval_json"]))
        fr["current"] = True
        frames.append(fr)
    params = [(n, tuple(p.shape), p.numel()) for n, p in trainer.model.named_parameters()]
    from dataclasses import asdict
    return ExportContext(
        hist=hist, meta=dict(trainer.data.meta), model_cfg=trainer.model_cfg.to_dict(), train_cfg=asdict(trainer.cfg),
        params=params, n_params=trainer.model.num_params(), n_params_non_emb=trainer.model.num_params(True),
        flops_per_token=trainer.flops_per_token, step=trainer.step, tokens=trainer.tokens,
        train_seconds=trainer.train_seconds, tokens_per_epoch=trainer.sampler.n_windows * trainer.model_cfg.block_size,
        dataset=trainer.data.name, fingerprint=trainer.data.fingerprint, tokenizer=trainer.data.tokenizer,
        frames=frames, checkpoint=checkpoint)


# --------------------------------------------------------------------------- #
# Options and helpers
# --------------------------------------------------------------------------- #
@dataclass
class ExportOptions:
    themes: tuple = ("print", "slides")
    width: str = "double"                        # single | onehalf | double (print theme)
    formats: tuple = ("png", "pdf", "svg")
    sections: tuple = ("A", "B", "C", "D", "E")
    max_strip: int = 6                           # frames per time-lapse strip


def _mpl():
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # the CJK fallback has no bold
    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    return matplotlib, plt, LinearSegmentedColormap


def heat_cmap():
    _, _, LSC = _mpl()
    return LSC.from_list("lab_heat", [tuple(c / 255 for c in rgb) for rgb in HEAT_ANCHORS])


def rc(theme: str) -> dict:
    t = THEMES[theme]
    return {
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "text.color": t["fg"], "axes.labelcolor": t["fg"], "axes.edgecolor": t["spine"], "axes.titlecolor": t["fg"],
        "xtick.color": t["spine"], "ytick.color": t["spine"], "xtick.labelcolor": t["dim"], "ytick.labelcolor": t["dim"],
        "grid.color": t["grid"], "axes.grid": True, "grid.linewidth": t["thin"], "axes.axisbelow": True,
        "font.family": ["Arial", "Helvetica", "Arial Unicode MS", "DejaVu Sans"],   # a list: per-glyph fallback
        "font.size": t["base"], "axes.titlesize": t["title"], "axes.labelsize": t["base"],
        "xtick.labelsize": t["base"] - 1, "ytick.labelsize": t["base"] - 1, "legend.fontsize": t["base"] - 1,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": t["thin"] + 0.1, "xtick.major.width": t["thin"], "ytick.major.width": t["thin"],
        "xtick.major.size": 2.5 if theme == "print" else 5, "ytick.major.size": 2.5 if theme == "print" else 5,
        "lines.linewidth": t["line"], "axes.titlelocation": "left", "axes.titleweight": "bold",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "image.interpolation": "nearest",
        "figure.constrained_layout.use": True,
    }


def fig_size(theme: str, width: str, aspect: float) -> tuple[float, float]:
    """Figure size in inches: print = journal column width x width / aspect; slides = 16:9."""
    if theme == "slides":
        return SLIDE_IN
    w = WIDTHS_MM.get(width, 183.0) / 25.4
    return (w, w / aspect)


def letter(ax, text: str, theme: str) -> None:
    t = THEMES[theme]
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(-5 if theme == "print" else -10, 4),
                textcoords="offset points",
                ha="right", va="bottom", fontsize=t["letter"], fontweight="bold", color=t["fg"], annotation_clip=False)


def safe(s: str) -> str:
    """Token text for matplotlib: no mathtext; the newline mark written \\n (in every font)."""
    return s.replace("$", r"\$").replace("⏎", r"\n")


def tok_label(tok, i: int) -> str:
    return safe(tok.display(int(i)))


def sci(x: float) -> str:
    if not x:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    return rf"${x / 10 ** e:.2f}\times10^{{{e}}}$"


def fmt_tok(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


def smooth(a, k: int):
    a = np.asarray(a, dtype=np.float64)
    if k <= 1 or len(a) < 2:
        return a
    c = np.cumsum(np.insert(a, 0, 0.0))
    idx = np.arange(1, len(a) + 1)
    lo = np.maximum(0, idx - k)
    return (c[idx] - c[lo]) / (idx - lo)


def pick(frames: list, k: int) -> list:
    if len(frames) <= k:
        return list(frames)
    idx = np.unique(np.linspace(0, len(frames) - 1, k).round().astype(int))
    return [frames[i] for i in idx]


class Registry:
    """What was written: becomes README.md (the appendix index) and manifest.json."""

    def __init__(self, root: str):
        self.root = root
        self.items: list[dict] = []

    def add(self, section: str, ident: str, title: str, legend: str, files: list[str], kind: str = "figure") -> None:
        self.items.append({"section": section, "id": ident, "title": title, "legend": legend, "kind": kind,
                           "files": [os.path.relpath(f, self.root) for f in files]})


def save(fig, base: str, theme: str, formats: tuple) -> list[str]:
    _, plt, _ = _mpl()
    os.makedirs(os.path.dirname(base), exist_ok=True)
    out = []
    for fmt in formats:
        path = f"{base}_{theme}.{fmt}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(path, dpi=THEMES[theme]["dpi"], facecolor=fig.get_facecolor())
        out.append(path)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# B: curves
# --------------------------------------------------------------------------- #
def ax_loss(ax, c: ExportContext, theme: str, ppl_axis: bool = True) -> str:
    t = THEMES[theme]
    h = c.hist
    toks = np.asarray(h["tokens"], dtype=np.float64)
    loss = np.asarray(h["loss"], dtype=np.float64)
    ev = h["eval"]
    bl = c.meta.get("baselines", {})
    if len(toks):
        ax.plot(toks / 1e6, loss, color=t["raw"], lw=t["thin"], label="train (per update)")
        ax.plot(toks / 1e6, smooth(loss, 50), color=t["train"], lw=t["line"], label="train (50-update mean)")
    if ev:
        ax.plot([r["tokens"] / 1e6 for r in ev], [r["loss"] for r in ev], "o-", color=t["val"], lw=t["line"],
                ms=2.2 if theme == "print" else 5, label="validation (all tokens)")
    for key, name in (("unigram_val", "unigram"), ("bigram_val", "bigram")):
        if key in bl:
            ax.axhline(bl[key], color=t["ref"], lw=t["thin"], ls=(0, (4, 3)))
            ax.annotate(f"{name} {bl[key]:.2f}", xy=(1, bl[key]), xycoords=("axes fraction", "data"),
                        xytext=(-2, 2), textcoords="offset points", ha="right", va="bottom", color=t["dim"],
                        fontsize=t["base"] - 1)
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("loss (nats / token)")
    ax.set_title("Loss")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.88))
    if ppl_axis:
        lo, hi = ax.get_ylim()
        nice = [v * 10 ** e for e in range(0, 6) for v in (1, 2, 5)]
        ticks = [v for v in nice if lo <= math.log(v) <= hi]
        sec = ax.secondary_yaxis("right", functions=(lambda v: np.exp(np.clip(v, -50, 50)),
                                                     lambda p: np.log(np.maximum(p, 1e-30))))
        sec.set_yticks(ticks, [fmt_tok(v) for v in ticks])
        sec.set_ylabel("perplexity")
        sec.tick_params(labelsize=t["base"] - 1)
    if ev:
        a, b = ev[0], ev[-1]
        return (f"validation loss fell from {a['loss']:.2f} to {b['loss']:.2f} nats/token (perplexity "
                f"{a['ppl']:,.0f} to {b['ppl']:,.1f}) over {b['tokens'] / 1e6:.1f}M training tokens; dashed lines: "
                f"unigram ({bl.get('unigram_val', float('nan')):.2f}) and interpolated bigram "
                f"({bl.get('bigram_val', float('nan')):.2f}) models fitted on the training split")
    return "training loss per update"


def ax_lr(ax, c: ExportContext, theme: str) -> str:
    from train import TrainConfig, lr_at
    t = THEMES[theme]
    cfg = TrainConfig(**{k: v for k, v in c.train_cfg.items() if k in TrainConfig.__dataclass_fields__})
    s = np.unique(np.linspace(0, cfg.max_steps - 1, 400).astype(int))
    lr = np.array([lr_at(cfg, int(v)) for v in s])
    ax.plot(s / 1e3, lr * 1e3, color=t["lr"], lw=t["thin"], alpha=0.5, label="planned")
    done = s <= c.step
    if done.sum() > 1:
        ax.plot(s[done] / 1e3, lr[done] * 1e3, color=t["lr"], lw=t["line"], label="completed")
    ax.plot([c.step / 1e3], [lr_at(cfg, c.step) * 1e3], "o", color=t["lr"], ms=3 if theme == "print" else 7)
    ax.set_xlabel("update (thousands)")
    ax.set_ylabel(r"learning rate ($\times 10^{-3}$)")
    ax.set_title("Learning rate")
    ax.legend(loc="upper right")
    return (f"AdamW learning rate: linear warm-up over {cfg.warmup_steps:,} updates, then {cfg.schedule} decay to "
            f"{cfg.min_lr_ratio:g}× the peak {cfg.lr:g} over {cfg.max_steps:,} updates; dot: update {c.step:,}")


def ax_grad(ax, c: ExportContext, theme: str) -> str:
    t = THEMES[theme]
    g = np.asarray(c.hist["grad_norm"], dtype=np.float64)
    st = np.asarray(c.hist["step"], dtype=np.float64)
    clip = c.train_cfg.get("grad_clip", 0)
    if len(g):
        ax.plot(st, g, color=t["raw"], lw=t["thin"])
        ax.plot(st, smooth(g, 25), color=t["train"], lw=t["line"], label="25-update mean")
        ax.set_ylim(0, max(float(np.percentile(g, 99)) * 1.15, clip * 1.25 if clip else 0, 1e-3))
    if clip:
        ax.axhline(clip, color=t["accent"], lw=t["thin"], ls=(0, (4, 3)), label=f"clip at {clip:g}")
    ax.set_xlabel("update")
    ax.set_ylabel("global gradient norm")
    ax.set_title("Gradient norm")
    ax.legend(loc="upper right")
    frac = float((g[-200:] > clip).mean()) if clip and len(g) else 0.0
    return f"global L2 norm of the gradient before clipping; clipped in {100 * frac:.0f}% of the last {min(200, len(g))} updates"


def ax_accuracy(ax, c: ExportContext, theme: str) -> str:
    t = THEMES[theme]
    ev = c.hist["eval"]
    x = [r["tokens"] / 1e6 for r in ev]
    ax.plot(x, [100 * r["top5"] for r in ev], "o-", color=t["top5"], ms=2 if theme == "print" else 5, label="top-5")
    ax.plot(x, [100 * r["top1"] for r in ev], "o-", color=t["top1"], ms=2 if theme == "print" else 5, label="top-1")
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Next-token accuracy")
    ax.legend(loc="lower right")
    if ev:
        return f"share of validation tokens in the model's top 1 / top 5: {100 * ev[-1]['top1']:.1f}% / {100 * ev[-1]['top5']:.1f}%"
    return ""


def ax_calibration(ax, c: ExportContext, theme: str) -> str:
    t = THEMES[theme]
    ev = [r for r in c.hist["eval"] if r.get("calib_acc") is not None]
    ax.plot([0, 1], [0, 1], color=t["ref"], lw=t["thin"], ls=(0, (4, 3)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence of the top prediction")
    ax.set_ylabel("accuracy")
    ax.set_title("Calibration")
    if not ev:
        return ""
    r = ev[-1]
    acc = np.asarray(r["calib_acc"], dtype=np.float64)
    conf = np.asarray(r["calib_conf"], dtype=np.float64)
    nb = len(acc)
    centres = (np.arange(nb) + 0.5) / nb
    ok = np.isfinite(acc)
    ax.bar(centres[ok], acc[ok], width=0.9 / nb, color=t["val"], alpha=0.75, label="accuracy")
    ax.plot(centres[ok], conf[ok], "_", color=t["lr"], ms=8 if theme == "print" else 18, mew=1.5, label="mean confidence")
    ax.legend(loc="upper left")
    return f"reliability diagram at update {r['step']:,} ({nb} confidence bins); expected calibration error {r['ece']:.4f}"


def ax_position(ax, c: ExportContext, theme: str):
    _, plt, _ = _mpl()
    t = THEMES[theme]
    ev = [r for r in c.hist["eval"] if r.get("pos_loss")]
    if not ev:
        return "", None
    sel = pick(ev[1:] if len(ev) > 2 else ev, 6)
    cmap = plt.get_cmap("viridis")
    lo, hi = sel[0]["tokens"], max(sel[-1]["tokens"], sel[0]["tokens"] + 1)
    for r in sel:
        y = smooth(np.asarray(r["pos_loss"], dtype=np.float64), 5)
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, color=cmap(0.15 + 0.75 * (r["tokens"] - lo) / (hi - lo)), lw=t["line"])
    ax.set_xscale("log")
    ax.set_xlabel("position in the context (tokens)")
    ax.set_ylabel("validation loss (nats)")
    ax.set_title("Loss by position")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo / 1e6, hi / 1e6))
    icl = ev[-1].get("icl", float("nan"))
    txt = (f"validation loss at each context position, averaged over windows, for {len(sel)} evaluations "
           f"(colour: training tokens)")
    if math.isfinite(icl):
        txt += f"; in-context score (loss at token 500 minus token 50) {icl:+.3f}"
    return txt, sm


def ax_frequency(ax, c: ExportContext, theme: str) -> str:
    from train import FREQ_BUCKETS
    _, plt, _ = _mpl()
    t = THEMES[theme]
    ev = [r for r in c.hist["eval"] if r.get("freq_loss")]
    names = ["top 10", "11–100", "101–1k", "1k–10k", "> 10k"][: len(FREQ_BUCKETS)]
    cmap = plt.get_cmap("viridis")
    if ev:
        x = [r["tokens"] / 1e6 for r in ev]
        fl = np.asarray([r["freq_loss"] for r in ev], dtype=np.float64)
        for k in range(fl.shape[1]):
            ax.plot(x, fl[:, k], color=cmap(0.1 + 0.8 * k / max(1, fl.shape[1] - 1)), lw=t["line"], label=names[k])
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("validation loss (nats)")
    ax.set_title("Loss by token frequency")
    ax.legend(loc="upper right", ncol=2)
    return "validation loss of target tokens grouped by their training-frequency rank"


def ax_heads_history(ax, c: ExportContext, theme: str) -> str:
    _, plt, _ = _mpl()
    t = THEMES[theme]
    pr = c.hist["probe"]
    if len(pr) < 1:
        return ""
    x = [r["tokens"] / 1e6 for r in pr]
    ind = np.asarray([np.max(np.asarray(r["induction"]), axis=1) for r in pr])
    cmap = plt.get_cmap("plasma")
    for l in range(ind.shape[1]):
        ax.plot(x, ind[:, l], color=cmap(0.1 + 0.75 * l / max(1, ind.shape[1] - 1)), lw=t["line"], label=f"layer {l + 1}")
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("max induction score")
    ax.set_title("Induction heads")
    ax.legend(loc="upper left", ncol=2)
    return (f"largest induction score per layer on repeated random sequences; latest maximum {float(ind[-1].max()):.3f}")


def ax_copying(ax, c: ExportContext, theme: str) -> str:
    t = THEMES[theme]
    pr = c.hist["probe"]
    x = [r["tokens"] / 1e6 for r in pr]
    ax.plot(x, [r["loss_first"] for r in pr], color=t["ref"], lw=t["line"], label="first pass")
    ax.plot(x, [r["loss_repeat"] for r in pr], color=t["lr"], lw=t["line"], label="repeat")
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("loss (nats / token)")
    ax.set_title("In-context copying")
    ax.legend(loc="upper left")
    if pr:
        return (f"loss on random token sequences and on their exact repeat; latest {pr[-1]['loss_first']:.2f} vs "
                f"{pr[-1]['loss_repeat']:.2f} nats")
    return ""


def curves(c: ExportContext, opt: ExportOptions, reg: Registry, out: str) -> None:
    _, plt, _ = _mpl()
    singles = [("B1_loss", "Training and validation loss", ax_loss, 1.6),
               ("B2_learning_rate", "Learning-rate schedule", ax_lr, 1.9),
               ("B3_gradient_norm", "Gradient norm", ax_grad, 1.9),
               ("B4_accuracy", "Next-token accuracy", ax_accuracy, 1.9),
               ("B5_calibration", "Calibration", ax_calibration, 1.3),
               ("B7_frequency", "Loss by token frequency", ax_frequency, 1.9),
               ("B8_induction", "Induction heads over training", ax_heads_history, 1.9),
               ("B9_copying", "In-context copying", ax_copying, 1.9)]
    for ident, title, fn, aspect in singles:
        files, legend = [], ""
        for theme in opt.themes:
            with plt.rc_context(rc(theme)):
                fig, ax = plt.subplots(figsize=fig_size(theme, "single" if theme == "print" else opt.width, aspect))
                legend = fn(ax, c, theme)
                files += save(fig, os.path.join(out, ident), theme, opt.formats)
        reg.add("B", ident, title, legend, files)
    files, legend = [], ""
    for theme in opt.themes:
        with plt.rc_context(rc(theme)):
            fig, ax = plt.subplots(figsize=fig_size(theme, "single" if theme == "print" else opt.width, 1.6))
            legend, sm = ax_position(ax, c, theme)
            if sm is not None:
                cb = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
                cb.set_label("training tokens (M)")
                cb.outline.set_visible(False)
            files += save(fig, os.path.join(out, "B6_loss_by_position"), theme, opt.formats)
    reg.add("B", "B6_loss_by_position", "Loss by context position", legend, files)
    def ax_loss_strip(ax, c, theme):
        return ax_loss(ax, c, theme, ppl_axis=False)

    strips = [("B0_strip_training_dynamics", "Training dynamics", [ax_loss_strip, ax_lr, ax_grad], 2.9),
              ("B0_strip_evaluation", "Evaluation", [ax_accuracy, ax_calibration, ax_frequency], 2.9),
              ("B0_strip_mechanisms", "Mechanisms over training", [ax_heads_history, ax_copying], 2.6)]
    for ident, title, fns, aspect in strips:
        files, legends = [], []
        for theme in opt.themes:
            with plt.rc_context(rc(theme)):
                fig, axes = plt.subplots(1, len(fns), figsize=fig_size(theme, opt.width, aspect))
                legends = []
                for k, (ax, fn) in enumerate(zip(np.atleast_1d(axes), fns)):
                    legends.append(f"({chr(97 + k)}) {fn(ax, c, theme)}")
                    letter(ax, chr(97 + k), theme)
                files += save(fig, os.path.join(out, ident), theme, opt.formats)
        reg.add("B", ident, title, "; ".join(legends) + ".", files)


# --------------------------------------------------------------------------- #
# C: maps
# --------------------------------------------------------------------------- #
def map_attention_grid(fig, gs_axes, fr: dict, c: ExportContext, theme: str):
    attn = fr["attn"].astype(np.float32)
    L, H, n, _ = attn.shape
    im = None
    for l in range(L):
        for h in range(H):
            ax = gs_axes[l][h]
            im = ax.imshow(np.sqrt(attn[l, h]), cmap="magma", vmin=0, vmax=1, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if l == 0:
                ax.set_title(f"H{h + 1}", loc="center", fontweight="normal", fontsize=THEMES[theme]["base"] - 1)
            if h == 0:
                ax.set_ylabel(f"L{l + 1}", rotation=0, ha="right", va="center", fontsize=THEMES[theme]["base"] - 1)
    return im


def heat_grid(ax, data, theme: str, cmap: str, vmin: float, vmax: float, annotate: bool = True, fmt="{:.2f}"):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    L, H = data.shape
    ax.set_xticks(range(H), [str(h + 1) for h in range(H)])
    ax.set_yticks(range(L), [f"L{l + 1}" for l in range(L)])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if annotate:
        _, plt, _ = _mpl()
        cm = plt.get_cmap(cmap)
        for l in range(L):
            for h in range(H):
                v = float(data[l, h])
                r, g, b, _ = cm((v - vmin) / (vmax - vmin) if vmax > vmin else 0)
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                ax.text(h, l, fmt.format(v), ha="center", va="center", color="black" if lum > 0.45 else "white",
                        fontsize=max(4.0, THEMES[theme]["base"] - 2.5))
    return im


def map_logit_lens(ax, fr: dict, c: ExportContext, theme: str, labels: bool = True, n: int = 16):
    p = fr["lens_p_true"][:, :n].astype(np.float32)
    rows = p.shape[0]
    im = ax.imshow(np.sqrt(p[::-1]), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    toks = fr["tokens_ids"]
    ax.set_yticks(range(rows), [("embed" if r == 0 else f"block {r}") for r in range(rows)][::-1])
    ax.set_xticks(range(p.shape[1]), [tok_label(c.tokenizer, toks[i])[:8] for i in range(p.shape[1])],
                  fontsize=max(4.0, THEMES[theme]["base"] - 2))
    ax.set_xlabel("input token (position 1 →)")
    ax.tick_params(length=0)
    ax.grid(False)
    if labels:
        top = fr["lens_top"][:, :n][::-1]
        for r in range(rows):
            for i in range(p.shape[1]):
                v = math.sqrt(float(p[::-1][r, i]))
                ax.text(i, r, tok_label(c.tokenizer, top[r, i])[:7], ha="center", va="center",
                        color="black" if v > 0.62 else "white", fontsize=max(3.5, THEMES[theme]["base"] - 3.2))
    return im


def colorbar(fig, im, ax, label: str, theme: str):
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015, aspect=25)
    cb.set_label(label)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=THEMES[theme]["base"] - 1.5)
    return cb


def embedding_ax(ax, fr: dict, c: ExportContext, theme: str, lims=None, labels: int = 14):
    coords = fr["emb_coords"].astype(np.float32)
    n = len(coords)
    rank_t = 1.0 - np.arange(n) / max(n - 1, 1)
    order = np.argsort(rank_t)                                   # frequent tokens drawn last (on top)
    s = 0.4 if theme == "print" else 2.0
    sc = ax.scatter(coords[order, 0], coords[order, 1], c=rank_t[order], cmap=heat_cmap(), s=s, lw=0,
                    alpha=0.85, rasterized=True, vmin=0, vmax=1)
    ids = fr["emb_ids"]
    t = THEMES[theme]
    for i in range(min(labels, n)):
        ax.annotate(tok_label(c.tokenizer, ids[i]), (coords[i, 0], coords[i, 1]), fontsize=t["base"] - 2,
                    color=t["fg"], xytext=(2, 2), textcoords="offset points")
    if lims is not None:
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
    ax.set_xlabel(f"PC1 ({100 * float(fr['emb_explained'][0]):.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * float(fr['emb_explained'][1]):.1f}%)")
    ax.grid(False)
    return sc


def align(frames: list[dict]) -> None:
    """Flip principal axes of each frame to agree with the next (signs are arbitrary)."""
    for a, b in zip(frames[::-1][1:], frames[::-1][:-1]):
        if "emb_coords" in a and "emb_coords" in b and len(a["emb_coords"]) == len(b["emb_coords"]):
            ca, cb = a["emb_coords"].astype(np.float32), b["emb_coords"].astype(np.float32)
            for j in range(min(ca.shape[1], 3)):
                if float((ca[:, j] * cb[:, j]).sum()) < 0:
                    ca[:, j] *= -1
            a["emb_coords"] = ca


def maps(c: ExportContext, opt: ExportOptions, reg: Registry, out: str) -> None:
    _, plt, _ = _mpl()
    frames = [f for f in c.frames if "attn" in f]
    if not frames:
        return
    last = frames[-1]
    step = last["step"]
    tag = f"step{step:07d}"
    # C1 every head
    files = []
    for theme in opt.themes:
        with plt.rc_context(rc(theme) | {"figure.constrained_layout.use": False}):
            L, H = last["attn"].shape[:2]
            fig, axes = plt.subplots(L, H, figsize=fig_size(theme, opt.width, H / L * 1.12), squeeze=False)
            fig.subplots_adjust(left=0.06, right=0.9, top=0.93, bottom=0.03, wspace=0.08, hspace=0.08)
            im = map_attention_grid(fig, axes, last, c, theme)
            cax = fig.add_axes([0.92, 0.2, 0.012, 0.6])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label("attention (square-root scale)")
            cb.outline.set_visible(False)
            files += save(fig, os.path.join(out, f"C1_attention_heads_{tag}"), theme, opt.formats)
    n = last["attn"].shape[-1]
    reg.add("C", "C1_attention_heads", "Attention of every head",
            f"attention probabilities softmax(QKᵀ/√d) of all {last['attn'].shape[0]} layers × {last['attn'].shape[1]} "
            f"heads on the first {n} tokens of a validation passage at update {step:,} (rows: queries, columns: keys; "
            f"square-root colour scale)", files)
    # C2 head scores
    if "induction" in last:
        files = []
        specs = [("induction", "Induction", "inferno"), ("previous", "Previous token", "viridis"),
                 ("duplicate", "Duplicate token", "cividis"), ("head_entropy", "Entropy (nats)", "magma")]
        for theme in opt.themes:
            with plt.rc_context(rc(theme)):
                fig, axes = plt.subplots(1, 4, figsize=fig_size(theme, opt.width, 3.6))
                for k, (ax, (key, title, cmap)) in enumerate(zip(axes, specs)):
                    d = np.asarray(last[key], dtype=np.float32)
                    vmax = max(float(d.max()), 0.3 if key == "induction" else 1e-6)
                    im = heat_grid(ax, d, theme, cmap, 0.0, vmax)
                    ax.set_title(title)
                    letter(ax, chr(97 + k), theme)
                    colorbar(fig, im, ax, "", theme)
                files += save(fig, os.path.join(out, f"C2_head_scores_{tag}"), theme, opt.formats)
        reg.add("C", "C2_head_scores", "Attention-head scores",
                f"(a) induction, (b) previous-token and (c) duplicate-token scores of every head, measured on "
                f"{int(last.get('probe_seq_len', 0))} random tokens repeated twice; (d) mean attention entropy on the "
                f"validation passage; update {step:,}", files)
    # C3 logit lens
    if "lens_p_true" in last:
        files = []
        for theme in opt.themes:
            with plt.rc_context(rc(theme)):
                fig, ax = plt.subplots(figsize=fig_size(theme, opt.width, 2.3))
                im = map_logit_lens(ax, last, c, theme)
                colorbar(fig, im, ax, "p(actual next token), sqrt", theme)
                ax.set_title("Logit lens")
                files += save(fig, os.path.join(out, f"C3_logit_lens_{tag}"), theme, opt.formats)
        reg.add("C", "C3_logit_lens", "Logit lens",
                f"each layer's residual stream decoded with the final norm and unembedding (nostalgebraist 2020) for "
                f"the first 16 positions of the passage; colour: probability of the actual next token; text: the "
                f"layer's top prediction; update {step:,}", files)
    # C4 layer health
    hh = c.hist.get("health", [])
    if hh:
        files = []
        names = list(hh[-1][1].keys())
        steps = [s for s, _ in hh]
        ratio = np.array([[math.log10(max(rec.get(nm, (0, 0, 1e-12))[2], 1e-12)) for _, rec in hh] for nm in names])
        for theme in opt.themes:
            with plt.rc_context(rc(theme)):
                fig, ax = plt.subplots(figsize=fig_size(theme, opt.width, 2.4))
                x0, x1 = (steps[0], steps[-1]) if steps[-1] > steps[0] else (steps[0] - 0.5, steps[0] + 0.5)
                im = ax.imshow(ratio, cmap="inferno", vmin=-5, vmax=-1, aspect="auto",
                               extent=(x0, x1, len(names) - 0.5, -0.5))
                ax.set_yticks(range(len(names)), names)
                ax.tick_params(axis="y", labelsize=max(4, THEMES[theme]["base"] - 2.5), length=0)
                ax.set_xlabel("update")
                ax.set_title("Update size per weight matrix")
                ax.grid(False)
                colorbar(fig, im, ax, r"$\log_{10}\,|\Delta W| \,/\, |W|$", theme)
                files += save(fig, os.path.join(out, "C4_layer_health"), theme, opt.formats)
        reg.add("C", "C4_layer_health", "Update size per weight matrix",
                f"relative update size ‖ΔW‖/‖W‖ of every weight matrix, measured every "
                f"{c.train_cfg.get('stats_interval', 10)} updates over {len(steps)} measurements", files)
    # C5 embeddings
    if "emb_coords" in last:
        files = []
        for theme in opt.themes:
            with plt.rc_context(rc(theme)):
                fig, ax = plt.subplots(figsize=fig_size(theme, "single" if theme == "print" else opt.width, 1.15))
                sc = embedding_ax(ax, last, c, theme)
                ax.set_title("Token embeddings")
                cb = colorbar(fig, sc, ax, "training-frequency rank", theme)
                cb.set_ticks([0, 1], labels=["rare", "frequent"])
                files += save(fig, os.path.join(out, f"C5_embeddings_{tag}"), theme, opt.formats)
        ev = last["emb_explained"]
        reg.add("C", "C5_embeddings", "Token embeddings",
                f"all {len(last['emb_coords']):,} token embeddings on the first two principal components of the "
                f"frequency-weighted embedding distribution ({100 * float(ev[0]):.1f}% and {100 * float(ev[1]):.1f}% of "
                f"the variance); colour: training-frequency rank; labels: the most frequent tokens; update {step:,}", files)
    # C6 inside one block
    if "block_ln1" in last:
        files = []
        L = last["block_ln1"].shape[0]
        layer = int(np.unravel_index(int(np.argmax(last["induction"])), last["induction"].shape)[0]) if "induction" in last else L - 1
        head = int(np.argmax(last["induction"][layer])) if "induction" in last else 0
        n_tok = last["block_ln1"].shape[1]
        r_in = last["resid"][layer][:n_tok].astype(np.float32)
        r_out = last["resid"][layer + 1][:n_tok].astype(np.float32)
        act = last["block_mlp_act"][layer].astype(np.float32)
        top_units = np.sort(np.argsort(-np.abs(act).mean(axis=0))[:96])
        panels = [("residual in", r_in, "RdBu_r"), ("RMSNorm", last["block_ln1"][layer].astype(np.float32), "RdBu_r"),
                  (f"head {head + 1} attention", last["attn"][layer, head, :n_tok, :n_tok].astype(np.float32), "magma"),
                  ("heads · V", last["block_heads"][layer].astype(np.float32), "RdBu_r"),
                  ("MLP hidden", act[:, top_units], "RdBu_r"), ("residual out", r_out, "RdBu_r")]
        for theme in opt.themes:
            with plt.rc_context(rc(theme)):
                fig, axes = plt.subplots(1, len(panels), figsize=fig_size(theme, opt.width, 4.2))
                for k, (ax, (title, d, cmap)) in enumerate(zip(axes, panels)):
                    if cmap == "magma":
                        im = ax.imshow(np.sqrt(np.clip(d, 0, None)), cmap=cmap, vmin=0, vmax=1, aspect="auto")
                    else:
                        m = float(np.abs(d).max()) or 1.0
                        im = ax.imshow(d, cmap=cmap, vmin=-m, vmax=m, aspect="auto")
                    ax.set_title(title, fontsize=THEMES[theme]["base"] - 0.5)
                    ax.set_xticks([])
                    ax.set_yticks([] if k else range(0, n_tok, 4))
                    ax.grid(False)
                    letter(ax, chr(97 + k), theme)
                axes[0].set_ylabel("token position")
                files += save(fig, os.path.join(out, f"C6_block{layer + 1}_internals_{tag}"), theme, opt.formats)
        reg.add("C", "C6_block_internals", f"Inside block {layer + 1}",
                f"activations of block {layer + 1} for the first {n_tok} tokens (rows) at update {step:,}: (a) residual "
                f"stream entering the block, (b) after RMSNorm, (c) attention of head {head + 1} (the block's strongest "
                f"induction head), (d) attention-weighted values of all heads, (e) the 96 most active of "
                f"{act.shape[1]:,} MLP hidden units, (f) residual stream leaving the block; diverging panels use each "
                f"panel's own symmetric scale", files)
    # time-lapse strips
    if len(frames) >= 2:
        timelapse(frames, c, opt, reg, out)


def timelapse(frames: list[dict], c: ExportContext, opt: ExportOptions, reg: Registry, out: str) -> None:
    _, plt, _ = _mpl()
    sel = pick(frames, opt.max_strip)
    k = len(sel)
    steps = ", ".join(f"{f['step']:,}" for f in sel)
    last = sel[-1]
    # attention of the strongest induction head (fixed head across frames, shared scale)
    if "induction" in last:
        l, h = np.unravel_index(int(np.argmax(last["induction"])), last["induction"].shape)
        files = []
        for theme in opt.themes:
            with plt.rc_context(rc(theme)):
                fig, axes = plt.subplots(1, k, figsize=fig_size(theme, opt.width, k * 1.05))
                for j, (ax, fr) in enumerate(zip(np.atleast_1d(axes), sel)):
                    im = ax.imshow(np.sqrt(fr["attn"][l, h].astype(np.float32)), cmap="magma", vmin=0, vmax=1)
                    ax.set_title(f"update {fr['step']:,}", loc="center", fontweight="normal")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.grid(False)
                    letter(ax, chr(97 + j), theme)
                colorbar(fig, im, list(np.atleast_1d(axes)), "attention (sqrt)", theme)
                files += save(fig, os.path.join(out, f"C7_timelapse_attention_L{l + 1}H{h + 1}"), theme, opt.formats)
        reg.add("C", "C7_timelapse_attention", f"Layer {l + 1}, head {h + 1} across training",
                f"attention pattern of layer {l + 1}, head {h + 1} (the strongest induction head at the last capture) on "
                f"the same validation passage at updates {steps}; one colour scale for all panels", files)
        files = []
        vmax = max(0.3, max(float(np.max(fr["induction"])) for fr in sel))
        for theme in opt.themes:
            with plt.rc_context(rc(theme)):
                fig, axes = plt.subplots(1, k, figsize=fig_size(theme, opt.width, k * 1.0))
                for j, (ax, fr) in enumerate(zip(np.atleast_1d(axes), sel)):
                    im = heat_grid(ax, np.asarray(fr["induction"], dtype=np.float32), theme, "inferno", 0.0, vmax,
                                   annotate=k <= 4)
                    ax.set_title(f"update {fr['step']:,}", loc="center", fontweight="normal")
                    letter(ax, chr(97 + j), theme)
                colorbar(fig, im, list(np.atleast_1d(axes)), "induction score", theme)
                files += save(fig, os.path.join(out, "C8_timelapse_induction_scores"), theme, opt.formats)
        reg.add("C", "C8_timelapse_induction", "Induction scores across training",
                f"induction score of every head at updates {steps}; shared colour scale 0–{vmax:.2f}", files)
    if "lens_p_true" in last:
        files = []
        for theme in opt.themes:
            with plt.rc_context(rc(theme)):
                fig, axes = plt.subplots(1, k, figsize=fig_size(theme, opt.width, k * 0.95), sharey=True)
                for j, (ax, fr) in enumerate(zip(np.atleast_1d(axes), sel)):
                    im = map_logit_lens(ax, fr, c, theme, labels=False, n=16)
                    ax.set_xticks([])
                    if j:
                        ax.tick_params(labelleft=False)
                    ax.set_title(f"update {fr['step']:,}", loc="center", fontweight="normal")
                    letter(ax, chr(97 + j), theme)
                colorbar(fig, im, list(np.atleast_1d(axes)), "p(actual next), sqrt", theme)
                files += save(fig, os.path.join(out, "C9_timelapse_logit_lens"), theme, opt.formats)
        reg.add("C", "C9_timelapse_logit_lens", "Logit lens across training",
                f"probability of the actual next token decoded from every layer for the first 16 positions of the same "
                f"passage at updates {steps}; shared colour scale", files)
    emb = [f for f in sel if "emb_coords" in f]
    if len(emb) >= 2:
        align(emb)
        allc = np.concatenate([f["emb_coords"][:, :2].astype(np.float32) for f in emb])
        lo, hi = np.percentile(allc, 0.5, axis=0), np.percentile(allc, 99.5, axis=0)
        pad = 0.06 * (hi - lo)
        lims = ((lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1]))
        files = []
        for theme in opt.themes:
            with plt.rc_context(rc(theme)):
                fig, axes = plt.subplots(1, len(emb), figsize=fig_size(theme, opt.width, len(emb) * 1.0))
                for j, (ax, fr) in enumerate(zip(np.atleast_1d(axes), emb)):
                    embedding_ax(ax, fr, c, theme, lims=lims, labels=6)
                    ax.set_title(f"update {fr['step']:,}", loc="center", fontweight="normal")
                    if j:
                        ax.set_ylabel("")
                    letter(ax, chr(97 + j), theme)
                files += save(fig, os.path.join(out, "C10_timelapse_embeddings"), theme, opt.formats)
        emb_steps = ", ".join(f"{fr['step']:,}" for fr in emb)
        reg.add("C", "C10_timelapse_embeddings", "Token embeddings across training",
                f"all token embeddings on each capture's first two frequency-weighted principal components at updates "
                f"{emb_steps}; axis signs aligned between captures, shared limits; colour: training-frequency rank "
                f"(blue rare, red frequent)", files)


# --------------------------------------------------------------------------- #
# D: data sheets
# --------------------------------------------------------------------------- #
def write_table(rows: list[list], header: list[str], base: str, title: str, opt: ExportOptions,
                max_rows: int = 18) -> list[str]:
    """CSV (all rows) + Markdown (all rows) + typeset table figures (up to max_rows rows)."""
    _, plt, _ = _mpl()
    os.makedirs(os.path.dirname(base), exist_ok=True)
    files = []
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    files.append(base + ".csv")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(f"**{title}**\n\n| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(v) for v in r) + " |\n")
    files.append(base + ".md")
    shown = rows
    if len(rows) > max_rows:
        idx = np.unique(np.linspace(0, len(rows) - 1, max_rows).round().astype(int))
        shown = [rows[i] for i in idx]
    for theme in opt.themes:
        t = THEMES[theme]
        with plt.rc_context(rc(theme) | {"axes.grid": False, "figure.constrained_layout.use": False}):
            row_in = 0.16 if theme == "print" else 0.42
            title_in, note_in = (0.28, 0.16) if theme == "print" else (0.7, 0.4)
            body_in = row_in * (len(shown) + 1)
            h_in = title_in + body_in + note_in
            w_in = WIDTHS_MM[opt.width] / 25.4 if theme == "print" else SLIDE_IN[0]
            if theme == "slides" and h_in > SLIDE_IN[1]:
                row_in = (SLIDE_IN[1] - title_in - note_in) / (len(shown) + 1)
                body_in, h_in = row_in * (len(shown) + 1), SLIDE_IN[1]
            fig = plt.figure(figsize=(w_in, h_in))
            fig.text(0.004, 1 - 0.06 / h_in, title, ha="left", va="top", fontsize=t["title"], fontweight="bold",
                     color=t["fg"])
            ax = fig.add_axes([0.0, note_in / h_in, 1.0, body_in / h_in])
            ax.axis("off")
            tbl = ax.table(cellText=[[str(v) for v in r] for r in shown], colLabels=header, bbox=[0, 0, 1, 1],
                           cellLoc="right", colLoc="right", edges="horizontal")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(t["base"] - (0.5 if theme == "print" else 1))
            for (r, col), cell in tbl.get_celld().items():
                cell.set_edgecolor(t["grid"] if r else t["spine"])
                cell.set_linewidth(t["thin"])
                cell.set_facecolor(t["bg"])
                cell.get_text().set_color(t["fg"] if r else t["dim"])
                if col == 0:
                    cell.get_text().set_ha("left")
                    cell._loc = "left"
                if r == 0:
                    cell.get_text().set_fontweight("bold")
            if len(rows) > len(shown):
                fig.text(0.004, 0.25 * note_in / h_in, f"{len(shown)} of {len(rows)} rows shown, evenly spaced; all rows "
                         f"in the CSV", color=t["faint"], fontsize=t["base"] - 1.5, va="bottom")
            files += save(fig, base, theme, tuple(f for f in opt.formats if f in ("png", "pdf", "svg")))
    return files


def sheets(c: ExportContext, opt: ExportOptions, reg: Registry, out: str) -> None:
    m = c.meta
    rows = []
    for s in ("train", "validation", "test"):
        st = m.get("splits", {}).get(s)
        if st:
            rows.append([s, f"{st['rows']:,}", f"{st['words']:,}", f"{st['bytes'] / 1e6:.1f}", f"{st['tokens']:,}",
                         f"{st['bytes'] / max(st['tokens'], 1):.2f}"])
    files = write_table(rows, ["split", "rows", "words", "MB", "tokens", "bytes/token"],
                        os.path.join(out, "D1_corpus"), f"Corpus: {c.dataset}", opt)
    reg.add("D", "D1_corpus", "Corpus statistics",
            "rows, words (whitespace words plus one end-of-line per line, the WikiText convention), UTF-8 megabytes, "
            "BPE tokens and bytes per token of every split", files, kind="table")
    bl = m.get("baselines", {})
    vs = m.get("splits", {}).get("validation", {})
    bpt = vs.get("tokens", 1) / max(vs.get("bytes", 1), 1) / math.log(2)
    ref_rows = []
    for name, key in (("uniform over the vocabulary", "uniform"), ("unigram (training frequencies)", "unigram_val"),
                      ("interpolated bigram", "bigram_val")):
        if key in bl:
            v = bl[key]
            ref_rows.append([name, f"{v:.4f}", f"{math.exp(min(v, 50)):,.1f}", f"{v * bpt:.4f}"])
    ev = c.hist["eval"]
    if ev:
        best = min(ev, key=lambda r: r["loss"])
        ref_rows.append([f"this model, best (update {best['step']:,})", f"{best['loss']:.4f}", f"{best['ppl']:,.1f}",
                         f"{best['bpb']:.4f}"])
    files = write_table(ref_rows, ["model", "nats/token", "perplexity", "bits/byte"],
                        os.path.join(out, "D2_reference_models"), "Validation cross-entropy against references", opt)
    reg.add("D", "D2_reference_models", "Reference models",
            "validation cross-entropy of reference models fitted on the training split and of this transformer", files,
            kind="table")
    rows = [[f"{r['step']:,}", fmt_tok(r["tokens"]), f"{r['loss']:.4f}", f"{r['ppl']:,.1f}", f"{r['bpb']:.4f}",
             (f"{r['word_ppl']:,.1f}" if r.get("word_ppl") is not None and math.isfinite(r["word_ppl"]) else "-"),
             f"{100 * r['top1']:.2f}", f"{100 * r['top5']:.2f}", f"{r['ece']:.4f}"] for r in ev]
    files = write_table(rows, ["update", "tokens", "loss", "perplexity", "bits/byte", "word ppl", "top-1 %", "top-5 %",
                               "ECE"], os.path.join(out, "D3_validation_metrics"), "Validation metrics during training", opt)
    reg.add("D", "D3_validation_metrics", "Validation metrics",
            f"every validation pass ({len(rows)}): loss, perplexity, bits per byte, word-normalized perplexity, "
            "top-1 / top-5 accuracy and expected calibration error, each over all validation target tokens", files,
            kind="table")
    groups: dict[str, int] = {}
    for name, _shape, n in c.params:
        if name.startswith("blocks."):
            parts = name.split(".")
            key = f"block {int(parts[1]) + 1} · {parts[2]}"
        else:
            key = name.split(".")[0]
        groups[key] = groups.get(key, 0) + n
    rows = [[k, f"{v:,}", f"{100 * v / max(c.n_params, 1):.2f}"] for k, v in groups.items()]
    rows.append(["total (tied weights once)", f"{c.n_params:,}", "100.00"])
    files = write_table(rows, ["component", "parameters", "%"], os.path.join(out, "D4_parameters"),
                        "Parameters by component", opt, max_rows=40)
    with open(os.path.join(out, "D4_parameters_by_tensor.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tensor", "shape", "parameters"])
        for name, shape, n in c.params:
            w.writerow([name, "x".join(str(s) for s in shape), n])
    files.append(os.path.join(out, "D4_parameters_by_tensor.csv"))
    reg.add("D", "D4_parameters", "Parameters",
            "parameter counts by component (tied input/output embedding counted once); per-tensor shapes in the CSV",
            files, kind="table")
    last = next((f for f in reversed(c.frames) if "induction" in f), None)
    if last is not None:
        L, H = last["induction"].shape
        rows = [[l + 1, h + 1, f"{last['induction'][l, h]:.4f}", f"{last['previous'][l, h]:.4f}",
                 f"{last['duplicate'][l, h]:.4f}", f"{last['head_entropy'][l, h]:.3f}"]
                for l in range(L) for h in range(H)]
        rows.sort(key=lambda r: -float(r[2]))
        files = write_table(rows, ["layer", "head", "induction", "previous", "duplicate", "entropy"],
                            os.path.join(out, "D5_head_scores"), f"Head scores at update {last['step']:,}", opt,
                            max_rows=12)
        reg.add("D", "D5_head_scores", "Head scores",
                f"every head's induction, previous-token and duplicate-token score and attention entropy at update "
                f"{last['step']:,}, sorted by induction score (top 12 typeset; all in the CSV)", files, kind="table")
    rows = [[k, str(v)] for k, v in c.model_cfg.items()] + [[k, str(v)] for k, v in c.train_cfg.items()
                                                             if k not in ("data_dir",)]
    files = write_table(rows, ["setting", "value"], os.path.join(out, "D6_hyperparameters"), "Hyperparameters", opt,
                        max_rows=60)
    reg.add("D", "D6_hyperparameters", "Hyperparameters", "model architecture and training configuration", files,
            kind="table")
    if c.frames:
        evs = c.hist["eval"]

        def val_at(tok):
            prior = [r for r in evs if r["tokens"] <= tok]
            return f"{prior[-1]['loss']:.4f}" if prior else "-"

        rows = [[f"{f['step']:,}", fmt_tok(f["tokens"]), val_at(f["tokens"]),
                 "current state" if f.get("current") else "captured"] for f in c.frames]
        files = write_table(rows, ["update", "tokens", "last val loss", "source"], os.path.join(out, "D7_frames"),
                            "Frames used for maps", opt, max_rows=30)
        reg.add("D", "D7_frames", "Frames", "capture steps used for the maps and time-lapse strips", files, kind="table")


# --------------------------------------------------------------------------- #
# A: data cards
# --------------------------------------------------------------------------- #
def cards(c: ExportContext, opt: ExportOptions, reg: Registry, out: str) -> None:
    _, plt, _ = _mpl()
    ev = c.hist["eval"]
    e = ev[-1] if ev else None
    total_flops = c.tokens * c.flops_per_token
    items = [
        ("validation loss", f"{e['loss']:.3f}" if e else "-", "nats per token"),
        ("perplexity", f"{e['ppl']:,.1f}" if e else "-", f"{c.model_cfg['vocab_size']:,}-token BPE"),
        ("bits per byte", f"{e['bpb']:.3f}" if e else "-", "tokenizer-independent"),
        ("word perplexity", f"{e['word_ppl']:,.0f}" if e and math.isfinite(e.get('word_ppl', float('nan'))) else "-",
         "WikiText word count, raw text"),
        ("top-1 / top-5", f"{100 * e['top1']:.1f} / {100 * e['top5']:.1f}%" if e else "-", "validation accuracy"),
        ("training tokens", fmt_tok(c.tokens), f"{c.tokens / max(c.tokens_per_epoch, 1):.3f} epochs"),
        ("updates", f"{c.step:,}", f"of {c.train_cfg.get('max_steps', 0):,} planned"),
        ("parameters", f"{c.n_params / 1e6:.2f}M", f"{c.n_params_non_emb / 1e6:.2f}M non-embedding"),
        ("training compute", sci(total_flops), "FLOP (6N + 12LdT per token)"),
        ("wall-clock", f"{c.train_seconds / 60:.1f} min", "optimizer steps only"),
        ("architecture", f"{c.model_cfg['n_layer']}×{c.model_cfg['n_head']}×{c.model_cfg['d_model']}",
         f"layers × heads × width, context {c.model_cfg['block_size']}"),
        ("data", c.dataset, f"fingerprint {c.fingerprint[:10]}"),
    ]
    files = []
    for theme in opt.themes:
        t = THEMES[theme]
        with plt.rc_context(rc(theme) | {"axes.grid": False, "figure.constrained_layout.use": False}):
            fig = plt.figure(figsize=fig_size(theme, opt.width, 3.0))
            cols = 4
            for i, (label, value, sub) in enumerate(items):
                r, col = divmod(i, cols)
                ax = fig.add_axes([0.01 + col * 0.2475, 0.99 - (r + 1) * 0.325, 0.235, 0.3])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor(t["card"])
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_color(t["grid"])
                    s.set_linewidth(t["thin"])
                ax.text(0.06, 0.8, label.upper(), transform=ax.transAxes, color=t["dim"], fontsize=t["base"] - 1.5,
                        va="center")
                ax.text(0.06, 0.45, value, transform=ax.transAxes, color=t["fg"], fontsize=t["base"] * 1.9,
                        va="center")
                ax.text(0.06, 0.15, sub, transform=ax.transAxes, color=t["faint"], fontsize=t["base"] - 2,
                        va="center")
            files += save(fig, os.path.join(out, "A1_run_card"), theme, opt.formats)
    reg.add("A", "A1_run_card", "Run card",
            f"key numbers of the run at update {c.step:,}: validation metrics over all validation tokens, training "
            f"budget, model size and compute", files)
    mc, tc = c.model_cfg, c.train_cfg
    src = c.meta.get("source", {})
    sp = c.meta.get("splits", {})
    model_card = f"""# Model card

**Model.** Decoder-only transformer language model (GPT family) trained from scratch: {mc['n_layer']} layers,
{mc['n_head']} heads, width {mc['d_model']}, context {mc['block_size']} tokens, {mc['pos']} position encoding,
{mc['norm']} normalisation, {mc['mlp']} MLP, tied input/output embeddings; {c.n_params:,} parameters
({c.n_params_non_emb:,} non-embedding).

**Training.** {c.tokens:,} tokens ({c.tokens / max(c.tokens_per_epoch, 1):.3f} epochs) of {c.dataset}, {c.step:,}
updates of {tc['batch_size']}×{tc['grad_accum']}×{mc['block_size']} tokens; AdamW (β = {tc['beta1']}, {tc['beta2']},
weight decay {tc['weight_decay']} on matrices), peak learning rate {tc['lr']}, {tc['warmup_steps']} warm-up updates,
{tc['schedule']} schedule over {tc['max_steps']:,} updates, gradient clipping at {tc['grad_clip']}; seed {tc['seed']}.
Training compute about {total_flops:.2e} FLOP.

**Evaluation.** {"Validation loss " + f"{e['loss']:.4f} nats/token, perplexity {e['ppl']:,.1f}, {e['bpb']:.4f} bits per byte, top-1 accuracy {100 * e['top1']:.2f}%, expected calibration error {e['ece']:.4f}, at update {e['step']:,}, over all {e['eval_tokens']:,} validation target tokens." if e else "Not evaluated yet."}

**Intended use.** Research and teaching about how transformer language models learn: training dynamics,
attention, interpretability measurements. Not for generating text that is relied upon.

**Limitations.** A small model trained briefly: generated text is often ungrammatical and factually wrong. It
reflects the content and biases of English Wikipedia's good and featured articles. Word-normalized perplexity is
computed on the raw text and is not the classic `<unk>`-based WikiText-103 benchmark number.
"""
    data_card = f"""# Data card

**Dataset.** {c.dataset}{(" (" + src.get('config', '') + ", " + src.get('repo', '') + ")") if 'repo' in src else ""}.
WikiText (Merity et al., 2016, "Pointer Sentinel Mixture Models") is a collection of English Wikipedia articles
verified as Good or Featured. The raw variant keeps original casing, punctuation and numbers without `<unk>`
substitution.

**Licence.** Creative Commons Attribution-ShareAlike (CC BY-SA), as Wikipedia text. Attribute Wikipedia
contributors and Merity et al. when redistributing derived material.

**Composition.**

| split | rows | words | UTF-8 MB | BPE tokens |
|---|---|---|---|---|
""" + "\n".join(f"| {s} | {sp[s]['rows']:,} | {sp[s]['words']:,} | {sp[s]['bytes'] / 1e6:.1f} | {sp[s]['tokens']:,} |"
                for s in ("train", "validation", "test") if s in sp) + """

Words follow the WikiText convention: whitespace-separated words plus one end-of-line per line.

**Integrity.** Every downloaded file was verified against the SHA-256 published by the source repository:
""" + "\n".join(f"- `{name}`: `{sha}`" for split_files in src.get("files", {}).values() for name, sha in split_files) + f"""

**Preprocessing.** Each split's text is its dataset rows joined as lines (empty rows are blank lines). A
{c.model_cfg['vocab_size']:,}-token byte-level BPE (GPT-4 split pattern) was learned from the training split only and
used to tokenize every split. Data fingerprint `{c.fingerprint}`.

**Uses here.** Training (train split), model selection and monitoring (validation split); the test split is
reserved for a final evaluation.
"""
    for name, text in (("A2_model_card.md", model_card), ("A3_data_card.md", data_card)):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            f.write(text)
    reg.add("A", "A2_model_card", "Model card", "architecture, training, evaluation, intended use and limitations",
            [os.path.join(out, "A2_model_card.md")], kind="document")
    reg.add("A", "A3_data_card", "Data card", "source, licence, composition, checksums and preprocessing of the corpus",
            [os.path.join(out, "A3_data_card.md")], kind="document")


# --------------------------------------------------------------------------- #
# E: dashboard frames
# --------------------------------------------------------------------------- #
def dashboard_strips(c: ExportContext, opt: ExportOptions, reg: Registry, out: str) -> None:
    """Per-panel time-lapse strips of the PNG frames captured from the dashboard."""
    _, plt, _ = _mpl()
    by_panel: dict[str, list[tuple[int, str]]] = {}
    for fr in c.frames:
        d = fr.get("panels_dir")
        if not d or not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".png"):
                by_panel.setdefault(name[:-4], []).append((fr["step"], os.path.join(d, name)))
    if not by_panel:
        return
    import matplotlib.image as mpimg
    os.makedirs(out, exist_ok=True)
    for panel, items in sorted(by_panel.items()):
        sel = pick(items, opt.max_strip)
        imgs = [mpimg.imread(p) for _, p in sel]
        h_px = max(i.shape[0] for i in imgs)
        w_px = sum(i.shape[1] for i in imgs)
        t = THEMES["slides"]
        with plt.rc_context(rc("slides") | {"figure.constrained_layout.use": False}):
            dpi = 150
            title_px = 38
            fig = plt.figure(figsize=((w_px + 12 * (len(imgs) + 1)) / dpi, (h_px + title_px + 16) / dpi), dpi=dpi)
            x = 12
            W = w_px + 12 * (len(imgs) + 1)
            Ht = h_px + title_px + 16
            for (step, _), img in zip(sel, imgs):
                ax = fig.add_axes([x / W, 8 / Ht, img.shape[1] / W, img.shape[0] / Ht])
                ax.imshow(img)
                ax.axis("off")
                fig.text((x + 2) / W, (Ht - 10) / Ht, f"update {step:,}", color=t["dim"], fontsize=9, va="top")
                x += img.shape[1] + 12
            path = os.path.join(out, f"E_{panel}_strip.png")
            fig.savefig(path, dpi=dpi, facecolor="#000000")
            plt.close(fig)
        tab, _, sec = panel.partition("__")
        reg.add("E", f"E_{panel}", f"Dashboard · {tab} · {sec}",
                f"the '{sec}' panel of the {tab} tab as rendered by the dashboard at updates "
                f"{', '.join(f'{s:,}' for s, _ in sel)}", [path])


# --------------------------------------------------------------------------- #
# Appendix
# --------------------------------------------------------------------------- #
def build_appendix(c: ExportContext, out_dir: str, opt: ExportOptions | None = None,
                   progress=None) -> str:
    """Write the appendix folder; returns the path of its README.md."""
    opt = opt or ExportOptions()
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    reg = Registry(out_dir)
    steps = [("A", cards), ("B", curves), ("C", maps), ("D", sheets), ("E", dashboard_strips)]
    for k, (sec, fn) in enumerate(steps):
        if sec not in opt.sections:
            continue
        if progress:
            progress(f"building {SECTIONS[sec]}", k / len(steps))
        fn(c, opt, reg, os.path.join(out_dir, SECTIONS[sec]))
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0, 1),
        "checkpoint": c.checkpoint, "step": c.step, "tokens": c.tokens, "dataset": c.dataset,
        "data_fingerprint": c.fingerprint, "model": c.model_cfg, "train": c.train_cfg,
        "frames": [f["step"] for f in c.frames], "options": {"themes": list(opt.themes), "width": opt.width,
                                                             "formats": list(opt.formats), "sections": list(opt.sections)},
        "items": reg.items,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    readme = os.path.join(out_dir, "README.md")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(f"# Appendix: transformer language model on {c.dataset}\n\n")
        f.write(f"Exported {manifest['created']} at update {c.step:,} ({c.tokens:,} training tokens). Figures: "
                f"{' and '.join(opt.themes)} themes; print figures are {WIDTHS_MM.get(opt.width, 183):g} mm wide "
                f"(single-panel figures 89 mm) at 300 dpi, slides 1920 × 1080. Formats: {', '.join(opt.formats)}. "
                f"Time-lapse frames: {', '.join(f'{s:,}' for s in manifest['frames']) or 'none'}.\n\n")
        names = {"A": "A. Data cards", "B": "B. Curves", "C": "C. Maps", "D": "D. Data sheets",
                 "E": "E. Dashboard frames"}
        for sec in ("A", "B", "C", "D", "E"):
            items = [i for i in reg.items if i["section"] == sec]
            if not items:
                continue
            f.write(f"## {names[sec]}\n\n")
            for i in items:
                kind = {"table": "Table", "document": "Document"}.get(i["kind"], "Figure")
                f.write(f"**{kind} {i['id'].split('_')[0]}. {i['title']}.** {i['legend'][:1].upper()}{i['legend'][1:]}"
                        f"{'' if i['legend'].endswith('.') else '.'}\n\n")
                f.write("Files: " + ", ".join(f"[{os.path.basename(p)}]({p})" for p in i["files"]) + "\n\n")
    if progress:
        progress("done", 1.0)
    return readme


def default_export_dir(checkpoint: str | None, step: int, root: str | None = None) -> str:
    stem = os.path.splitext(os.path.basename(checkpoint or "run"))[0]
    base = root or os.path.join(os.path.dirname(os.path.abspath(checkpoint)) if checkpoint else ".", "exports")
    return os.path.join(base, f"appendix-{stem}-step{step:07d}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")


def default_frames_dir(checkpoint: str | None, root: str | None = None) -> str:
    stem = os.path.splitext(os.path.basename(checkpoint or "run"))[0]
    base = root or os.path.join(os.path.dirname(os.path.abspath(checkpoint)) if checkpoint else ".", "exports")
    return os.path.join(base, f"frames-{stem}")


def remove_tree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
