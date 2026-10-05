#!/usr/bin/env python3
"""Summarize frozen-checkpoint evaluations across independent training seeds.

Example:
  python summarize_evaluations.py --condition rope eval/rope_0 eval/rope_1 eval/rope_2 \
    --condition learned eval/learned_0 eval/learned_1 eval/learned_2 --split validation --out-dir report

Each directory holds the summary.json written by evaluate_checkpoint.py. All evaluations
in a comparison must use the same split, stride and corpus (checked). One evaluation per
training seed per condition; the bootstrap resamples training seeds, never tokens, because
tokens of one run are not independent replicates of the training process.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BOOTSTRAP_SAMPLES = 10000
METRICS = ("loss", "ppl", "bpb", "word_ppl", "top1", "top5", "ece", "icl")


def statistics(values: np.ndarray, rng: np.random.Generator) -> dict:
    out = {"n_training_seeds": len(values), "mean": float(values.mean()), "sample_sd": None,
           "bootstrap_95_low": None, "bootstrap_95_high": None}
    if len(values) > 1:
        out["sample_sd"] = float(values.std(ddof=1))
        idx = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
        lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
        out.update(bootstrap_95_low=float(lo), bootstrap_95_high=float(hi))
    return out


def load(directory: str, split: str) -> dict:
    path = Path(directory) / "summary.json"
    with path.open(encoding="utf-8") as f:
        s = json.load(f)
    if split not in s["splits"]:
        raise SystemExit(f"{path} has no {split!r} evaluation")
    return s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", nargs="+", action="append", required=True, metavar="NAME_OR_DIR",
                   help="condition name followed by evaluation directories")
    p.add_argument("--split", default="validation")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.bootstrap_seed)
    conditions: dict[str, list[dict]] = {}
    for group in args.condition:
        if len(group) < 2:
            raise SystemExit("--condition needs a name and at least one evaluation directory")
        name, dirs = group[0], group[1:]
        if name in conditions:
            raise SystemExit(f"duplicate condition {name!r}")
        conditions[name] = [load(d, args.split) for d in dirs]
    all_runs = [s for runs in conditions.values() for s in runs]
    for key in ("stride", "data_fingerprint"):
        if len({s[key] for s in all_runs}) > 1:
            raise SystemExit(f"evaluations differ in {key}; compare like with like")
    if len({s["splits"][args.split]["eval_tokens"] for s in all_runs}) > 1:
        raise SystemExit("evaluations scored different numbers of tokens")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, per_run = [], []
    for name, runs in conditions.items():
        seen = set()
        for s in runs:
            if s["checkpoint_sha256"] in seen:
                raise SystemExit(f"{name}: the same checkpoint appears twice")
            seen.add(s["checkpoint_sha256"])
            per_run.append({"condition": name, "checkpoint": s["checkpoint"], "step": s["step"],
                            "tokens": s["tokens"], **{m: s["splits"][args.split].get(m) for m in METRICS}})
        for m in METRICS:
            vals = np.array([s["splits"][args.split].get(m) for s in runs], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                rows.append({"condition": name, "metric": m, **statistics(vals, rng)})
    with (args.out_dir / "per_run.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0]))
        w.writeheader()
        w.writerows(per_run)
    with (args.out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"split": args.split, "rows": rows, "runs": per_run}, f, indent=2)
    for r in rows:
        if r["metric"] in ("loss", "bpb", "word_ppl"):
            ci = (f"  95% bootstrap over seeds [{r['bootstrap_95_low']:.4f}, {r['bootstrap_95_high']:.4f}]"
                  if r["bootstrap_95_low"] is not None else "  (one seed: no interval)")
            sd = f" sd {r['sample_sd']:.4f}" if r["sample_sd"] is not None else ""
            print(f"{r['condition']:<16} {r['metric']:<9} mean {r['mean']:.4f}{sd} n={r['n_training_seeds']}{ci}")
    print(f"wrote {args.out_dir}/summary.csv, per_run.csv, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
