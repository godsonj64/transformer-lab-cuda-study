"""Analysis for the paper study: RoPE vs learned absolute positions.

Reads only what the training runs logged in their checkpoints (`history`): per-head
induction and previous-token scores every 25 updates, the repeated-random-token losses,
and validation evaluations with loss by context position. Writes CSV tables, a JSON of
every number quoted in the paper, and the paper figures.

Usage: python paper_study/analyze.py [--runs rope_s0 learned_s0 ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt")
OUT = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "paper", "figures")

TH_IND = 0.2      # induction-score threshold for the threshold-crossing definition
TH_PREV = 0.5     # previous-token-score threshold
TH_COPY = 1.0     # nats: repeated-sequence loss below first-copy loss


def pwlf3(x: np.ndarray, y: np.ndarray, min_gap: int = 3) -> dict:
    """Continuous three-segment piecewise-linear least-squares fit with knots k1 < k2 at
    observed x values (exhaustive search). Following Aoyama et al. (2025, App. C.1), the
    first knot is the emergence point."""
    n = len(x)
    best = None
    for i in range(min_gap, n - 2 * min_gap):
        for j in range(i + min_gap, n - min_gap):
            k1, k2 = x[i], x[j]
            A = np.stack([np.ones(n), x, np.maximum(x - k1, 0), np.maximum(x - k2, 0)], 1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            sse = float(np.sum((A @ coef - y) ** 2))
            if best is None or sse < best["sse"]:
                best = {"k1": float(k1), "k2": float(k2), "sse": sse, "coef": coef.tolist()}
    return best


def first_crossing(steps: np.ndarray, vals: np.ndarray, th: float, k: int = 3) -> float:
    """First step at which `vals` is >= th for k consecutive probes (nan if never)."""
    ok = vals >= th
    for i in range(len(ok) - k + 1):
        if ok[i:i + k].all():
            return float(steps[i])
    return float("nan")


def load_run(name: str) -> dict:
    ck = torch.load(os.path.join(CKPT, f"{name}.pt"), map_location="cpu", weights_only=False)
    h = ck["history"]
    pr = {}
    for p in h["probe"]:          # a resumed run can log a step twice; keep the last
        pr[int(p["step"])] = p
    steps = np.array(sorted(pr))
    ind = np.array([np.array(pr[s]["induction"]) for s in steps])      # (n, L, H)
    prev = np.array([np.array(pr[s]["previous"]) for s in steps])
    lf = np.array([pr[s]["loss_first"] for s in steps])
    lr_ = np.array([pr[s]["loss_repeat"] for s in steps])
    ev = {}
    for e in h["eval"]:
        if e.get("split") == "validation" and e.get("full", True):
            ev[int(e["step"])] = e
    esteps = np.array(sorted(ev))
    from model import GPTConfig, GPT
    mcfg = GPTConfig(**ck["model_config"])
    m = GPT(mcfg)
    return {
        "name": name, "pos": ck["model_config"]["pos"], "seed": ck["train_config"]["seed"],
        "step": int(ck["step"]), "tokens": int(ck["tokens"]), "train_seconds": float(ck["train_seconds"]),
        "params": m.num_params(), "params_ne": m.num_params(True),
        "tokens_per_step": ck["train_config"]["batch_size"] * ck["train_config"].get("grad_accum", 1) * mcfg.block_size,
        "steps": steps, "ind": ind, "prev": prev, "loss_first": lf, "loss_repeat": lr_,
        "esteps": esteps, "eval_loss": np.array([ev[s]["loss"] for s in esteps]),
        "eval_bpb": np.array([ev[s]["bpb"] for s in esteps]),
        "eval_icl": np.array([ev[s]["icl"] for s in esteps]),
        "eval_top1": np.array([ev[s]["top1"] for s in esteps]),
        "pos_loss": np.array([ev[s]["pos_loss"] for s in esteps], dtype=float),
        "train_loss": np.array(h["loss"], dtype=float), "train_step": np.array(h["step"]),
        "lr": np.array(h["lr"], dtype=float),
    }


def summarize(r: dict) -> dict:
    s = r["steps"]
    ind_max = r["ind"].reshape(len(s), -1).max(1)
    prev_max = r["prev"].reshape(len(s), -1).max(1)
    copy = r["loss_first"] - r["loss_repeat"]
    fit = pwlf3(s.astype(float), ind_max)
    L, H = r["ind"].shape[1:]
    fi, fp = r["ind"][-1], r["prev"][-1]
    li, hi = np.unravel_index(np.argmax(fi), fi.shape)
    lp, hp = np.unravel_index(np.argmax(fp), fp.shape)
    tps = r["tokens_per_step"]
    # ICL score at the last evaluation before and the first after the induction emergence point
    out = {
        "name": r["name"], "pos": r["pos"], "seed": r["seed"], "final_step": r["step"], "final_tokens": r["tokens"],
        "train_minutes": r["train_seconds"] / 60, "params": r["params"], "params_non_embedding": r["params_ne"],
        "ind_emerge_pwlf_step": fit["k1"], "ind_plateau_pwlf_step": fit["k2"],
        "ind_emerge_pwlf_Mtok": fit["k1"] * tps / 1e6,
        "ind_cross_step": first_crossing(s, ind_max, TH_IND),
        "prev_cross_step": first_crossing(s, prev_max, TH_PREV),
        "copy_cross_step": first_crossing(s, copy, TH_COPY),
        "icl_cross_step": first_crossing(r["esteps"], -r["eval_icl"], 0.0, k=2),
        "final_ind_max": float(ind_max[-1]), "final_prev_max": float(prev_max[-1]),
        "final_copy_nats": float(copy[-1]),
        "final_ind_layer": int(li), "final_ind_head": int(hi), "final_prev_layer": int(lp), "final_prev_head": int(hp),
        "n_heads_ind_gt_0.2": int((fi > TH_IND).sum()),
        "final_val_loss": float(r["eval_loss"][-1]), "final_val_bpb": float(r["eval_bpb"][-1]),
        "final_val_top1": float(r["eval_top1"][-1]), "final_icl": float(r["eval_icl"][-1]),
        "max_icl": float(np.nanmax(r["eval_icl"])), "max_icl_step": int(r["esteps"][int(np.nanargmax(r["eval_icl"]))]),
        "eval_step_final": int(r["esteps"][-1]),
    }
    # validation loss at the eval step closest to the RoPE and learned emergence windows (matched tokens)
    for st in (1000, 1500, 2000, 2500, 3000):
        k = np.where(r["esteps"] == st)[0]
        out[f"val_loss_at_{st}"] = float(r["eval_loss"][k[0]]) if len(k) else float("nan")
        k = np.where(r["esteps"] == st)[0]
        out[f"icl_at_{st}"] = float(r["eval_icl"][k[0]]) if len(k) else float("nan")
    return out


def fig_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
                         "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0,
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    return plt


COL = {"rope": "#0072B2", "learned": "#D55E00"}     # Okabe-Ito
LS = {0: "-", 1: "--", 2: ":"}
LAB = {"rope": "RoPE", "learned": "Learned abs."}


def figures(runs: list[dict], sums: list[dict]) -> None:
    plt = fig_style()
    os.makedirs(FIG, exist_ok=True)
    # Figure 1: induction max, previous max, copying, ICL vs step
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.4))
    for r, sm in zip(runs, sums):
        s = r["steps"]
        c, ls = COL[r["pos"]], LS.get(r["seed"], "-")
        lab = f"{LAB[r['pos']]}, seed {r['seed']}"
        ax[0, 0].plot(s, r["ind"].reshape(len(s), -1).max(1), color=c, ls=ls, label=lab)
        ax[0, 0].axvline(sm["ind_emerge_pwlf_step"], color=c, ls=ls, lw=0.5, alpha=0.6)
        ax[0, 1].plot(s, r["prev"].reshape(len(s), -1).max(1), color=c, ls=ls, label=lab)
        ax[1, 0].plot(s, r["loss_first"] - r["loss_repeat"], color=c, ls=ls, label=lab)
        ax[1, 1].plot(r["esteps"], r["eval_icl"], color=c, ls=ls, marker=".", ms=2, label=lab)
    ax[0, 0].set(title="(a) Best induction score (max over 36 heads)", xlabel="update", ylabel="attention to induction target")
    ax[0, 1].set(title="(b) Best previous-token score", xlabel="update", ylabel="attention to previous token")
    ax[1, 0].set(title="(c) In-context copying on repeated random tokens", xlabel="update",
                 ylabel="loss(1st copy) − loss(repeat), nats")
    ax[1, 1].set(title="(d) ICL score on validation (Olsson et al.)", xlabel="update",
                 ylabel="loss@500 − loss@50, nats")
    ax[1, 1].axhline(0, color="0.5", lw=0.5)
    ax[0, 0].legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dynamics.pdf"))
    plt.close(fig)

    # Figure 2: validation loss and train loss
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
    for r in runs:
        c, ls = COL[r["pos"]], LS.get(r["seed"], "-")
        lab = f"{LAB[r['pos']]}, seed {r['seed']}"
        ax[0].plot(r["esteps"][1:], r["eval_loss"][1:], color=c, ls=ls, label=lab)
        k = 50
        tl = np.convolve(r["train_loss"], np.ones(k) / k, mode="valid")
        ax[1].plot(r["train_step"][k - 1:], tl, color=c, ls=ls, label=lab)
    ax[0].set(title="(a) Validation loss (all 263,950 tokens)", xlabel="update", ylabel="nats/token", ylim=(3.9, 6.5))
    ax[1].set(title="(b) Training loss (50-update mean)", xlabel="update", ylabel="nats/token", ylim=(3.9, 6.5))
    ax[0].legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_loss.pdf"))
    plt.close(fig)

    # Figure 3: loss by context position, before vs after emergence, one seed per condition
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
    for a, pos in zip(ax, ("rope", "learned")):
        rr = [r for r in runs if r["pos"] == pos]
        if not rr:
            continue
        r = rr[0]
        cmap = plt.get_cmap("viridis")
        sel = [i for i, st in enumerate(r["esteps"]) if st in (500, 1000, 1500, 2000, 2500, 3000)]
        for j, i in enumerate(sel):
            pl = r["pos_loss"][i]
            x = np.arange(1, len(pl) + 1)
            k = 9
            sm = np.convolve(pl, np.ones(k) / k, mode="valid")
            a.plot(x[k // 2: k // 2 + len(sm)], sm, color=cmap(j / max(1, len(sel) - 1)), label=f"update {r['esteps'][i]}")
        a.set(title=f"({'a' if pos == 'rope' else 'b'}) {LAB[pos]}, seed {r['seed']}: loss by position",
              xlabel="context position (tokens)", xscale="log")
        a.legend(frameon=False, ncol=2)
    ax[0].set_ylabel("validation loss, nats/token")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_position.pdf"))
    plt.close(fig)

    # Figure 4: per-layer induction / previous scores at the final step, mean over seeds
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.9))
    for col, pos in enumerate(("rope", "learned")):
        rr = [r for r in runs if r["pos"] == pos]
        if not rr:
            continue
        ind = np.mean([r["ind"][-1] for r in rr], 0)
        prv = np.mean([r["prev"][-1] for r in rr], 0)
        for k, (mat, nm) in enumerate(((prv, "previous-token"), (ind, "induction"))):
            a = ax[2 * col + k]
            im = a.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            a.set(title=f"{LAB[pos]}: {nm}", xlabel="head", ylabel="layer" if k == 0 and col == 0 else None)
            a.set_xticks(range(mat.shape[1]))
            a.set_yticks(range(mat.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.savefig(os.path.join(FIG, "fig_heads.pdf"))
    plt.close(fig)


def main(argv=None) -> int:
    sys.path.insert(0, os.path.join(HERE, ".."))
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--no-fig", action="store_true")
    a = p.parse_args(argv)
    names = a.runs or sorted(os.path.basename(f)[:-3] for f in glob.glob(os.path.join(CKPT, "*.pt"))
                             if os.path.exists(os.path.join(HERE, "done_" + os.path.basename(f)[:-3])))
    os.makedirs(OUT, exist_ok=True)
    runs = [load_run(n) for n in names]
    sums = [summarize(r) for r in runs]
    import csv
    with open(os.path.join(OUT, "per_run.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sums[0]))
        w.writeheader()
        w.writerows(sums)
    # per-run trajectories
    for r in runs:
        with open(os.path.join(OUT, f"traj_{r['name']}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "induction_max", "previous_max", "loss_first", "loss_repeat"])
            for i, st in enumerate(r["steps"]):
                w.writerow([int(st), float(r["ind"][i].max()), float(r["prev"][i].max()),
                            float(r["loss_first"][i]), float(r["loss_repeat"][i])])
    # condition summaries and paired (same seed) differences
    agg = {}
    keys = [k for k, v in sums[0].items() if isinstance(v, (int, float)) and k not in ("seed",)]
    for pos in ("rope", "learned"):
        ss = [s for s in sums if s["pos"] == pos]
        if ss:
            agg[pos] = {k: {"mean": float(np.nanmean([s[k] for s in ss])),
                            "sd": float(np.nanstd([s[k] for s in ss], ddof=1)) if len(ss) > 1 else float("nan"),
                            "values": [s[k] for s in ss], "seeds": [s["seed"] for s in ss]} for k in keys}
    paired = {}
    seeds = sorted({s["seed"] for s in sums})
    for k in keys:
        d = []
        for sd in seeds:
            a_ = [s for s in sums if s["pos"] == "learned" and s["seed"] == sd]
            b_ = [s for s in sums if s["pos"] == "rope" and s["seed"] == sd]
            if a_ and b_:
                d.append(a_[0][k] - b_[0][k])
        if d:
            paired[k] = {"learned_minus_rope": d, "mean": float(np.mean(d)),
                         "sd": float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"per_run": sums, "by_condition": agg, "paired": paired}, f, indent=1)
    for s in sums:
        print(f"{s['name']:<12} emerge(pwlf) {s['ind_emerge_pwlf_step']:>6.0f}  cross0.2 {s['ind_cross_step']:>6.0f}  "
              f"prev0.5 {s['prev_cross_step']:>5.0f}  copy1 {s['copy_cross_step']:>6.0f}  icl<0 {s['icl_cross_step']:>6.0f}  "
              f"ind {s['final_ind_max']:.3f} L{s['final_ind_layer']}H{s['final_ind_head']}  prev {s['final_prev_max']:.3f} "
              f"L{s['final_prev_layer']}H{s['final_prev_head']}  val {s['final_val_loss']:.4f}  icl {s['final_icl']:+.3f}")
    if not a.no_fig:
        figures(runs, sums)
    return 0


if __name__ == "__main__":
    sys.exit(main())
