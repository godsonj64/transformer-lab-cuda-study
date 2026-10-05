#!/usr/bin/env python3
"""Evaluate a frozen checkpoint on WikiText validation and/or test.

Example:
  python evaluate_checkpoint.py --checkpoint gpt_wikitext.pt --split validation --stride 256 \
    --out-dir eval/baseline_0

Every target token of the split is scored exactly once with sliding windows: each window
holds block_size tokens and, after the first, scores only its last `stride` tokens, so every
scored token has at least block_size - stride tokens of context (stride = block_size gives
non-overlapping windows, the cheaper estimate used during training). Reported: loss in nats
per token, perplexity, bits per byte (independent of the tokenizer), word-level perplexity
with the WikiText word count (whitespace words + one end-of-line per line), top-1 / top-5
accuracy, expected calibration error, and loss by context position.

The weights are never updated. Use the test split once, for the final report; choose
settings on validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

from data import TokenData, prepare
from model import GPTConfig, Sampler
from train import TrainConfig, Trainer, load_checkpoint, resolve_device


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="validation", help="validation, test or validation,test")
    p.add_argument("--stride", type=int, default=None, help="tokens scored per window (default: block_size // 2)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--samples", type=int, default=3, help="generated samples to store (0 = none)")
    p.add_argument("--sample-tokens", type=int, default=80)
    p.add_argument("--out-dir", default=None, help="writes summary.json here (default: print only)")
    args = p.parse_args(argv)
    try:
        ck = load_checkpoint(args.checkpoint)
        mcfg = GPTConfig(**ck["model_config"])
        tcfg = TrainConfig(**{k: v for k, v in ck["train_config"].items() if k in TrainConfig.__dataclass_fields__})
        info = prepare(tcfg.dataset, args.data_dir, mcfg.vocab_size)
        data = TokenData(info)
        device = resolve_device(args.device)
        tcfg.device = device.type
        tr = Trainer(mcfg, tcfg, data, device=device)
        tr.load_state(ck)
    except (ValueError, OSError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    stride = args.stride or max(1, mcfg.block_size // 2)
    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    result = {
        "checkpoint": os.path.abspath(args.checkpoint), "checkpoint_sha256": file_sha256(args.checkpoint),
        "step": tr.step, "tokens": tr.tokens, "train_seconds": tr.train_seconds, "dataset": data.name,
        "data_fingerprint": data.fingerprint, "model": mcfg.to_dict(), "params": tr.model.num_params(),
        "params_non_embedding": tr.model.num_params(True), "stride": stride, "device": device.type,
        "torch": torch.__version__, "splits": {},
    }
    print(f"{args.checkpoint}: step {tr.step:,}, {tr.tokens / 1e6:.1f}M training tokens, "
          f"{tr.model.num_params() / 1e6:.2f}M parameters; context {mcfg.block_size}, stride {stride}")
    for split in splits:
        t0 = time.time()
        r = tr.evaluate(split, stride=stride, batch=args.batch_size)
        r = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}
        r["seconds"] = round(time.time() - t0, 1)
        r["words"] = data.meta["splits"][split]["words"]
        r["bytes"] = data.meta["splits"][split]["bytes"]
        result["splits"][split] = r
        wp = f"{r['word_ppl']:,.2f}" if math.isfinite(r["word_ppl"]) else "-"
        print(f"  {split:<10} loss {r['loss']:.4f} nats/token · ppl {r['ppl']:,.2f} · {r['bpb']:.4f} bits/byte · "
              f"word ppl {wp} · top-1 {r['top1']:.2%} · top-5 {r['top5']:.2%} · ECE {r['ece']:.4f} "
              f"({r['eval_tokens']:,} tokens, {r['seconds']} s)")
    bl = data.meta.get("baselines", {})
    if bl:
        result["baselines_validation"] = bl
        print(f"  references on validation: unigram {bl.get('unigram_val', float('nan')):.4f}, "
              f"bigram {bl.get('bigram_val', float('nan')):.4f} nats/token")
    samples = []
    for i in range(args.samples):
        prompt = (" = = History = = \n", " The film was", " In 1916 ,")[i % 3]
        s = Sampler(tr.model, data.tokenizer.encode(prompt), temperature=0.8, top_p=0.95, seed=i)
        text = data.tokenizer.decode([s.step()["token"] for _ in range(args.sample_tokens)])
        samples.append({"prompt": prompt, "seed": i, "temperature": 0.8, "top_p": 0.95, "text": text})
    result["samples"] = samples
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
