"""Final-checkpoint measurements for the paper study (no training).

For each final checkpoint:
  1. head scores on a larger, shared probe set (64 random sequences of 48 tokens, each
     repeated twice; the same sequences for every run);
  2. zero-ablation of single heads (the head's output is set to zero before W_o):
     the copying gain loss(first copy) - loss(repeat) after ablating each of the 36 heads,
     and the validation loss on a fixed subset of windows after ablating the top
     induction head and the top previous-token head.

Usage: python paper_study/final_probes.py [--device cpu] [--runs ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from model import GPT, GPTConfig  # noqa: E402
from probes import head_scores  # noqa: E402
from train import load_checkpoint  # noqa: E402

PROBE_SEED = 10_000
PROBE_BATCH = 64
VAL_WINDOWS = 64          # 64 x 512 = 32,768 validation tokens, fixed offsets


class Ablate:
    """Zero the output of the given (layer, head) pairs before the output projection."""

    def __init__(self, model: GPT, heads: list[tuple[int, int]]):
        self.model, self.heads, self.handles = model, heads, []

    def __enter__(self):
        hd = self.model.cfg.head_dim
        by_layer: dict[int, list[int]] = {}
        for l, h in self.heads:
            by_layer.setdefault(l, []).append(h)
        for l, hs in by_layer.items():
            def pre(mod, args, hs=hs):
                y = args[0].clone()
                for h in hs:
                    y[..., h * hd:(h + 1) * hd] = 0
                return (y,)
            self.handles.append(self.model.blocks[l].attn.proj.register_forward_pre_hook(pre))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def copy_losses(model: GPT, x: torch.Tensor, S: int) -> tuple[float, float]:
    logits, _ = model(x)
    V = logits.shape[-1]
    nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, V), x[:, 1:].reshape(-1), reduction="none").view(x.shape[0], -1)
    return float(nll[:, :S - 1].mean()), float(nll[:, S - 1:].mean())


@torch.no_grad()
def val_loss(model: GPT, windows: torch.Tensor) -> tuple[float, np.ndarray]:
    tot, pos = 0.0, None
    for i in range(0, len(windows), 16):
        w = windows[i:i + 16]
        logits, _ = model(w[:, :-1])
        V = logits.shape[-1]
        nll = F.cross_entropy(logits.float().reshape(-1, V), w[:, 1:].reshape(-1), reduction="none").view(w.shape[0], -1)
        tot += float(nll.sum())
        pos = nll.sum(0) if pos is None else pos + nll.sum(0)
    n = windows.shape[0] * (windows.shape[1] - 1)
    return tot / n, (pos / windows.shape[0]).cpu().numpy()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--runs", nargs="*", default=None)
    a = p.parse_args(argv)
    names = a.runs or sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(HERE, "ckpt", "*.pt"))
                             if os.path.exists(os.path.join(HERE, "done_" + os.path.basename(f)[:-3])))
    dev = torch.device(a.device)
    val = np.memmap(os.path.join(ROOT, "data", "wikitext-103", "validation-16384.bin"), dtype=np.uint16, mode="r")
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    allres = {}
    for name in names:
        ck = load_checkpoint(os.path.join(HERE, "ckpt", f"{name}.pt"))
        cfg = GPTConfig(**ck["model_config"])
        model = GPT(cfg)
        model.load_state_dict(ck["model"])
        model.to(dev).eval()
        T = cfg.block_size
        offs = np.linspace(0, len(val) - T - 2, VAL_WINDOWS).astype(np.int64)
        windows = torch.tensor(np.stack([np.asarray(val[o:o + T + 1], dtype=np.int64) for o in offs]), device=dev)

        hs = head_scores(model, seq_len=48, batch=PROBE_BATCH, seed=PROBE_SEED)
        S = hs["seq_len"]
        ind, prev, dup = hs["induction"], hs["previous"], hs["duplicate"]
        rng = np.random.default_rng(PROBE_SEED)
        r = rng.integers(256, cfg.vocab_size, size=(PROBE_BATCH, S))
        x = torch.tensor(np.concatenate([r, r], 1), device=dev)
        base_first, base_rep = copy_losses(model, x, S)
        base_val, base_pos = val_loss(model, windows)
        L, H = ind.shape
        li, hi = map(int, np.unravel_index(np.argmax(ind), ind.shape))
        lp, hp = map(int, np.unravel_index(np.argmax(prev), prev.shape))
        # every single-head ablation: copying gain
        gain = np.zeros((L, H))
        for l in range(L):
            for h in range(H):
                with Ablate(model, [(l, h)]):
                    f_, r_ = copy_losses(model, x, S)
                gain[l, h] = f_ - r_
        base_gain = base_first - base_rep
        res = {"name": name, "pos": cfg.pos, "seed": ck["train_config"]["seed"], "step": int(ck["step"]),
               "induction": ind.tolist(), "previous": prev.tolist(), "duplicate": dup.tolist(),
               "top_ind": [li, hi], "top_ind_score": float(ind[li, hi]),
               "top_prev": [lp, hp], "top_prev_score": float(prev[lp, hp]),
               "loss_first": base_first, "loss_repeat": base_rep, "copy_gain": base_gain,
               "ablate_gain": gain.tolist(),
               "gain_drop_top_ind": base_gain - float(gain[li, hi]),
               "gain_drop_top_prev": base_gain - float(gain[lp, hp]),
               "gain_drop_rank_top_ind": int((base_gain - gain > base_gain - gain[li, hi]).sum()) + 1,
               "gain_drop_rank_top_prev": int((base_gain - gain > base_gain - gain[lp, hp]).sum()) + 1,
               "gain_drop_median_other": float(np.median(np.delete((base_gain - gain).ravel(), [li * H + hi, lp * H + hp]))),
               "val_loss_subset": base_val, "val_tokens_subset": int(windows.shape[0] * (windows.shape[1] - 1))}
        # induction attention after removing the top previous-token head (does the circuit compose?)
        with Ablate(model, [(lp, hp)]):
            hs2 = head_scores(model, seq_len=48, batch=PROBE_BATCH, seed=PROBE_SEED)
            v2, pos2 = val_loss(model, windows)
        res["top_ind_score_after_prev_ablation"] = float(hs2["induction"][li, hi])
        res["max_ind_after_prev_ablation"] = float(hs2["induction"].max())
        res["val_loss_ablate_prev"] = v2
        with Ablate(model, [(li, hi)]):
            v1, pos1 = val_loss(model, windows)
        res["val_loss_ablate_ind"] = v1
        # loss by position change (late minus early context) under each ablation
        def late_early(pl):
            return float(np.mean(pl[400:510]) - np.mean(pl[40:60]))
        res["late_minus_early_base"] = late_early(base_pos)
        res["late_minus_early_ablate_ind"] = late_early(pos1)
        res["late_minus_early_ablate_prev"] = late_early(pos2)
        allres[name] = res
        print(f"{name:<12} top ind L{li}H{hi} {ind[li, hi]:.3f} | top prev L{lp}H{hp} {prev[lp, hp]:.3f} | "
              f"copy gain {base_gain:.3f} -> -ind {gain[li, hi]:.3f} (rank {res['gain_drop_rank_top_ind']}), "
              f"-prev {gain[lp, hp]:.3f} (rank {res['gain_drop_rank_top_prev']}), median other drop "
              f"{res['gain_drop_median_other']:.3f} | ind after -prev {res['top_ind_score_after_prev_ablation']:.3f} | "
              f"val {base_val:.4f} -ind {v1:.4f} -prev {v2:.4f}")
    with open(os.path.join(out_dir, "final_probes.json"), "w") as f:
        json.dump(allres, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
