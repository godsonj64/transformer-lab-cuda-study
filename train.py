"""
train.py - training, evaluation, checkpoints and logs for the transformer language model.

Training objective: next-token cross-entropy, L = -1/N sum_t log p(x_{t+1} | x_<=t), on
windows of block_size tokens.

Optimiser: AdamW (Loshchilov & Hutter 2019), betas (0.9, 0.95), decoupled weight decay on
weight matrices only, global gradient-norm clipping, optional gradient accumulation.
Learning rate: linear warm-up, then cosine decay to min_lr_ratio x peak at max_steps (or
"wsd": warm-up, constant, linear decay over the last decay_frac of the steps; or constant).
The schedule is a function of the step number, so it continues exactly after a resume.

Pause, resume, continue
    Everything that determines the future of a run is in the checkpoint: weights, AdamW
    moments, step and token counters, the configuration, the tokenizer, the data
    fingerprint and the full metric history. Batches are a pure function of (seed, step),
    so a resumed run sees exactly the batches it would have seen without the interruption
    (bit-for-bit identical weights on the CPU; GPU kernels are not bit-deterministic).
    Checkpoints are written to a temporary file and renamed, so an interrupted save never
    destroys the previous one.

Evaluation reports, on every target token of the validation split:
    loss (nats/token), perplexity exp(loss), bits per byte (tokenizer-independent),
    word-level perplexity (WikiText convention), top-1/top-5 accuracy, expected calibration
    error, loss by context position (and the in-context-learning score of Olsson et al.:
    loss at token 500 minus loss at token 50), and loss by token frequency.
"""
from __future__ import annotations

import csv
import json
import math
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from bpe import Tokenizer
from data import BatchSampler, TokenData, eval_windows, gather
from model import GPT, GPTConfig, Sampler
from probes import embedding_analysis, head_scores, snippet_analysis

CHECKPOINT_FORMAT = "transformer_lab_v1"
SAMPLE_PROMPTS = (" = = Early life = = \n He was born in", " The city is located on the")
FREQ_BUCKETS = ((0, 10), (10, 100), (100, 1000), (1000, 10000), (10000, 10 ** 9))


@dataclass
class TrainConfig:
    dataset: str = "wikitext-103"
    data_dir: str = "data"
    batch_size: int = 16               # sequences per micro-batch
    grad_accum: int = 1                # micro-batches per optimizer step
    max_steps: int = 20000             # schedule length; training stops here (can be extended)
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    schedule: str = "cosine"           # cosine | wsd | constant
    decay_frac: float = 0.2            # wsd: final fraction of steps spent decaying
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0             # 0 = no clipping
    eval_interval: int = 250           # steps between validation passes (0 = never)
    eval_tokens: int = 0               # 0 = the whole validation split
    probe_interval: int = 50           # steps between attention / lens / head-score snapshots
    sample_interval: int = 500         # steps between generated samples
    stats_interval: int = 10           # steps between layer-health measurements
    seed: int = 1337
    device: str = "auto"
    dtype: str = "auto"                # auto | float32 | bfloat16 (autocast)

    def validate(self) -> None:
        if self.batch_size < 1 or self.grad_accum < 1 or self.max_steps < 1:
            raise ValueError("batch_size, grad_accum and max_steps must be >= 1")
        if self.schedule not in ("cosine", "wsd", "constant"):
            raise ValueError("schedule must be cosine, wsd or constant")
        if not (self.lr > 0 and 0 <= self.min_lr_ratio <= 1 and 0 < self.decay_frac <= 1):
            raise ValueError("lr must be > 0, min_lr_ratio in [0, 1], decay_frac in (0, 1]")
        if self.dtype not in ("auto", "float32", "bfloat16"):
            raise ValueError("dtype must be auto, float32 or bfloat16")


def lr_at(cfg: TrainConfig, step: int) -> float:
    """Learning rate for optimizer step `step` (0-based)."""
    peak, low = cfg.lr, cfg.lr * cfg.min_lr_ratio
    if step < cfg.warmup_steps:
        return peak * (step + 1) / cfg.warmup_steps
    if cfg.schedule == "constant":
        return peak
    if cfg.schedule == "wsd":
        start = int(cfg.max_steps * (1 - cfg.decay_frac))
        if step < start:
            return peak
        frac = min(1.0, (step - start) / max(1, cfg.max_steps - start))
        return peak + (low - peak) * frac
    span = max(1, cfg.max_steps - cfg.warmup_steps)
    frac = min(1.0, (step - cfg.warmup_steps) / span)
    return low + 0.5 * (peak - low) * (1 + math.cos(math.pi * frac))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS (Apple GPU) is not available")
    return torch.device(name)


def autocast_dtype(name: str, device: torch.device):
    if name == "bfloat16" or (name == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class RunLogger:
    """CSV logs of one session in runs/<timestamp>/."""

    FILES = {
        "metrics.csv": ["step", "tokens", "loss", "lr", "grad_norm", "tokens_per_s", "train_seconds"],
        "eval.csv": ["step", "tokens", "split", "loss", "ppl", "bpb", "word_ppl", "top1", "top5", "ece", "icl",
                     "eval_tokens"],
        "probes.csv": ["step", "tokens", "induction_max", "previous_max", "loss_first", "loss_repeat"],
    }

    def __init__(self, directory: str, config: dict):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        self._files, self._writers = {}, {}
        for name, header in self.FILES.items():
            path = os.path.join(directory, name)
            fresh = not os.path.exists(path)
            fh = open(path, "a", newline="", encoding="utf-8")
            w = csv.writer(fh)
            if fresh:
                w.writerow(header)
            self._files[name], self._writers[name] = fh, w
        self._samples = open(os.path.join(directory, "samples.txt"), "a", encoding="utf-8")
        self._last_flush = time.time()

    def write(self, name: str, row: list) -> None:
        self._writers[name].writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
        if time.time() - self._last_flush > 5:
            self.flush()

    def sample(self, step: int, prompt: str, text: str) -> None:
        self._samples.write(f"--- step {step} ---\n{prompt}{text}\n\n")

    def flush(self) -> None:
        for fh in self._files.values():
            fh.flush()
        self._samples.flush()
        self._last_flush = time.time()

    def close(self) -> None:
        self.flush()
        for fh in self._files.values():
            fh.close()
        self._samples.close()


def default_run_dir(base: str = "runs") -> str:
    return os.path.join(base, datetime.now().strftime("%Y%m%d-%H%M%S"))


class Trainer:
    """One training run: model, optimizer, data order, periodic measurements and history."""

    def __init__(self, model_cfg: GPTConfig, cfg: TrainConfig, data: TokenData,
                 logger: RunLogger | None = None, device: torch.device | None = None):
        cfg.validate()
        if model_cfg.vocab_size != data.vocab_size:
            raise ValueError(f"model vocab {model_cfg.vocab_size} != tokenizer vocab {data.vocab_size}")
        self.cfg, self.model_cfg, self.data, self.logger = cfg, model_cfg, data, logger
        self.device = device or resolve_device(cfg.device)
        self.amp = autocast_dtype(cfg.dtype, self.device)
        torch.manual_seed(cfg.seed)
        self.model = GPT(model_cfg).to(self.device)
        self.opt = torch.optim.AdamW(self.model.param_groups(cfg.weight_decay), lr=cfg.lr,
                                     betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
        self.sampler = BatchSampler(len(data["train"]), model_cfg.block_size, cfg.seed)
        self.step = 0
        self.tokens = 0
        self.train_seconds = 0.0
        self.hist: dict[str, list] = {"step": [], "tokens": [], "loss": [], "lr": [], "grad_norm": [],
                                      "tok_s": [], "eval": [], "probe": [], "health": [], "samples": []}
        self.last_eval: dict | None = None
        self.last_probe: dict | None = None
        self.snapshot: dict | None = None
        self.embed: dict | None = None
        self.last_batch: dict | None = None
        self.log: list[tuple[str, str]] = []
        self.snippet_start = self.paragraph_start(int(np.random.default_rng(cfg.seed).integers(0, 1 << 30)))
        self.snippet_len = min(model_cfg.block_size, 256) + 1
        self.flops_per_token = self.model.flops_per_token()
        self.stop_reason: str | None = None
        self.on_task = None                       # callback(name) before eval / probe / sample

    # -- bookkeeping ------------------------------------------------------------- #
    def add_log(self, kind: str, msg: str) -> None:
        self.log.append((kind, msg))
        del self.log[:-400]

    @property
    def tokens_per_step(self) -> int:
        return self.cfg.batch_size * self.cfg.grad_accum * self.model_cfg.block_size

    @property
    def epoch(self) -> float:
        return self.sampler.epoch_of(self.step * self.cfg.batch_size * self.cfg.grad_accum)

    @property
    def done(self) -> bool:
        return self.step >= self.cfg.max_steps

    def lr(self, step: int | None = None) -> float:
        return lr_at(self.cfg, self.step if step is None else step)

    def batch(self, step: int, micro: int) -> tuple[torch.Tensor, torch.Tensor]:
        B = self.cfg.batch_size
        first = (step * self.cfg.grad_accum + micro) * B
        starts = self.sampler.starts(first, B)
        x, y = gather(self.data["train"], starts, self.model_cfg.block_size)
        return self.to_device(x), self.to_device(y)

    def to_device(self, a: np.ndarray) -> torch.Tensor:
        # Blocking copy on purpose: with torch 2.12, a non_blocking host-to-MPS copy can deliver
        # stale memory instead of the batch (reproduced with strided arrays; see the tests).
        t = torch.from_numpy(np.ascontiguousarray(a))
        if self.device.type == "cuda":
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    def _autocast(self):
        if self.amp is None:
            return torch.autocast(self.device.type, enabled=False)
        return torch.autocast(self.device.type, dtype=self.amp)

    # -- one optimizer step -------------------------------------------------------- #
    def train_step(self) -> dict:
        cfg, model = self.cfg, self.model
        model.train()
        lr = self.lr()
        for g in self.opt.param_groups:
            g["lr"] = lr
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=self.device)
        keep = self.step % cfg.stats_interval == 0
        for micro in range(cfg.grad_accum):
            x, y = self.batch(self.step, micro)
            with self._autocast():
                logits, loss = model(x, y)
            (loss / cfg.grad_accum).backward()
            loss_sum += loss.detach()
            if keep and micro == 0:
                tok_nll = F.cross_entropy(logits[0].detach().float(), y[0], reduction="none")
                self.last_batch = {"step": self.step, "x": x[0].cpu().numpy(), "y": y[0].cpu().numpy(),
                                   "nll": tok_nll.cpu().numpy()}
            del logits, loss
        params = [p for g in self.opt.param_groups for p in g["params"]]
        if cfg.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        else:
            grad_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params if p.grad is not None]))
        health = None
        if keep:
            groups = model.tensor_groups()
            before = [[p.detach().clone() for p in ps] for _, ps in groups]
            grads = [math.sqrt(sum(float(p.grad.pow(2).sum()) for p in ps) / sum(p.numel() for p in ps))
                     for _, ps in groups]
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        if keep:
            health = {}
            for (name, ps), old, g in zip(groups, before, grads):
                w = math.sqrt(sum(float(p.detach().pow(2).sum()) for p in ps) / sum(p.numel() for p in ps))
                upd = math.sqrt(sum(float((p.detach() - o).pow(2).sum()) for p, o in zip(ps, old)))
                wn = math.sqrt(sum(float(o.pow(2).sum()) for o in old))
                health[name] = (w, g, upd / max(wn, 1e-12))
        loss_v = float(loss_sum) / cfg.grad_accum                       # synchronises the device
        gn = float(grad_norm)
        dt = time.perf_counter() - t0
        self.step += 1
        self.tokens += self.tokens_per_step
        self.train_seconds += dt
        tok_s = self.tokens_per_step / max(dt, 1e-9)
        h = self.hist
        h["step"].append(self.step)
        h["tokens"].append(self.tokens)
        h["loss"].append(loss_v)
        h["lr"].append(lr)
        h["grad_norm"].append(gn)
        h["tok_s"].append(tok_s)
        if health is not None:
            h["health"].append((self.step, health))
        if self.logger:
            self.logger.write("metrics.csv", [self.step, self.tokens, loss_v, lr, gn, tok_s, self.train_seconds])
        if not math.isfinite(loss_v):
            self.stop_reason = f"loss became {loss_v} at step {self.step}; lower --lr or check the data"
            self.add_log("warn", self.stop_reason)
        return {"loss": loss_v, "lr": lr, "grad_norm": gn, "tok_s": tok_s, "seconds": dt}

    def advance(self) -> dict | None:
        """One optimizer step followed by any measurement that is due. None when finished."""
        if self.stop_reason or self.done:
            return None
        self.initial_measurements()
        st = self.train_step()
        self.periodic()
        if self.done:
            self.add_log("info", f"reached max_steps {self.cfg.max_steps:,}; raise --max-steps to continue")
        return st

    def eval_due(self, step: int) -> bool:
        """Every eval_interval steps, and five times as often during the first interval
        (the loss changes fastest at the start)."""
        n = self.cfg.eval_interval
        if not n:
            return False
        return step % n == 0 or (step < n and step % max(1, n // 5) == 0)

    def initial_measurements(self) -> None:
        """The starting point (before any update, or after a resume) of every chart."""
        if not self.hist["eval"] and self.cfg.eval_interval:
            self._task("eval")
            self.run_eval()
        if self.last_probe is None:
            self._task("probe")
            self.run_probes()
        if not self.hist["samples"] and self.cfg.sample_interval:
            self._task("sample")
            self.run_sample()
        self._task("")

    def periodic(self) -> None:
        c, s = self.cfg, self.step
        if self.eval_due(s):
            self._task("eval")
            self.run_eval()
        if c.probe_interval and s % c.probe_interval == 0:
            self._task("probe")
            self.run_probes()
        if c.sample_interval and s % c.sample_interval == 0:
            self._task("sample")
            self.run_sample()
        self._task("")

    def _task(self, name: str) -> None:
        if self.on_task is not None:
            self.on_task(name)

    # -- evaluation ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, split: str = "validation", max_tokens: int = 0, stride: int | None = None,
                 batch: int | None = None) -> dict:
        """Score every target token of a split (or an evenly spaced subset of windows when
        max_tokens > 0) with sliding windows; see data.eval_windows."""
        model, T = self.model, self.model_cfg.block_size
        arr = self.data[split]
        plan = eval_windows(len(arr), T, stride)
        full = True
        if max_tokens and max_tokens < len(arr):
            n = max(1, max_tokens // T)
            if n < len(plan):
                pick = np.unique(np.linspace(0, len(plan) - 1, n).round().astype(int))
                plan = [plan[i] for i in pick]
                full = False
        B = batch or max(1, self.cfg.batch_size)
        was = model.training
        model.eval()
        tot_nll = 0.0
        n_tok = 0
        n_bytes = 0
        pos_sum = np.zeros(T)
        pos_cnt = np.zeros(T)
        top1 = top5 = 0
        bins = 15
        conf_sum, acc_sum, bin_cnt = np.zeros(bins), np.zeros(bins), np.zeros(bins)
        counts = self.data.unigram
        rank_of = None
        if counts is not None:
            order = np.argsort(-counts, kind="stable")
            rank_of = np.empty(len(order), dtype=np.int64)
            rank_of[order] = np.arange(len(order))
        fb_sum, fb_cnt = np.zeros(len(FREQ_BUCKETS)), np.zeros(len(FREQ_BUCKETS))
        token_len = self.data.token_len
        groups: dict[int, list] = {}
        for w in plan:
            groups.setdefault(w[1], []).append(w)
        try:
            for length, ws in groups.items():
                for i in range(0, len(ws), B):
                    chunk = ws[i:i + B]
                    starts = np.array([w[0] for w in chunk], dtype=np.int64)
                    x, y = gather(arr, starts, length)
                    xt, yt = self.to_device(x), self.to_device(y)
                    with self._autocast():
                        logits, _ = model(xt)
                    logp = logits.float().log_softmax(-1)
                    nll = -logp.gather(-1, yt[..., None])[..., 0]
                    top = logp.topk(5, dim=-1).indices
                    hit1 = (top[..., 0] == yt)
                    hit5 = (top == yt[..., None]).any(-1)
                    conf = logp.max(-1).values.exp()
                    nll_c, hit1_c, hit5_c, conf_c = (t.cpu().numpy() for t in (nll, hit1, hit5, conf))
                    first = np.array([w[2] for w in chunk])
                    mask = np.arange(length)[None, :] >= first[:, None]
                    tot_nll += float(nll_c[mask].astype(np.float64).sum())
                    n_tok += int(mask.sum())
                    n_bytes += int(token_len[y][mask].sum())
                    top1 += int(hit1_c[mask].sum())
                    top5 += int(hit5_c[mask].sum())
                    b = np.clip((conf_c[mask] * bins).astype(int), 0, bins - 1)
                    np.add.at(conf_sum, b, conf_c[mask])
                    np.add.at(acc_sum, b, hit1_c[mask])
                    np.add.at(bin_cnt, b, 1)
                    if length == T:
                        full_rows = first == 0
                        if full_rows.any():
                            pos_sum += nll_c[full_rows].sum(axis=0)
                            pos_cnt += full_rows.sum()
                    if rank_of is not None:
                        r = rank_of[y[mask]]
                        for k, (lo, hi) in enumerate(FREQ_BUCKETS):
                            m = (r >= lo) & (r < hi)
                            fb_sum[k] += float(nll_c[mask][m].sum())
                            fb_cnt[k] += int(m.sum())
        finally:
            model.train(was)
        loss = tot_nll / max(n_tok, 1)
        pos_loss = np.divide(pos_sum, pos_cnt, out=np.full(T, np.nan), where=pos_cnt > 0)
        ece = float(np.sum(np.abs(acc_sum - conf_sum)) / max(n_tok, 1))
        res = {"split": split, "step": self.step, "tokens": self.tokens, "loss": loss, "ppl": math.exp(min(loss, 50)),
               "bpb": tot_nll / math.log(2) / max(n_bytes, 1), "top1": top1 / max(n_tok, 1), "top5": top5 / max(n_tok, 1),
               "ece": ece, "eval_tokens": n_tok, "full": full,
               "calib_conf": np.divide(conf_sum, bin_cnt, out=np.full(bins, np.nan), where=bin_cnt > 0),
               "calib_acc": np.divide(acc_sum, bin_cnt, out=np.full(bins, np.nan), where=bin_cnt > 0),
               "calib_n": bin_cnt, "pos_loss": pos_loss,
               "freq_loss": np.divide(fb_sum, fb_cnt, out=np.full(len(fb_cnt), np.nan), where=fb_cnt > 0)}
        words = self.data.meta.get("splits", {}).get(split, {}).get("words", 0)
        res["word_ppl"] = math.exp(min(tot_nll / words, 50)) if full and words else float("nan")
        res["icl"] = icl_score(pos_loss)
        return res

    def run_eval(self) -> dict:
        t0 = time.perf_counter()
        r = self.evaluate("validation", self.cfg.eval_tokens)
        r["seconds"] = time.perf_counter() - t0
        self.last_eval = r
        self.hist["eval"].append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()})
        if self.logger:
            self.logger.write("eval.csv", [self.step, self.tokens, "validation", r["loss"], r["ppl"], r["bpb"],
                                           r["word_ppl"], r["top1"], r["top5"], r["ece"], r["icl"], r["eval_tokens"]])
        wp = f", word ppl {r['word_ppl']:.1f}" if math.isfinite(r["word_ppl"]) else ""
        self.add_log("eval", f"step {self.step:,}: val loss {r['loss']:.4f}, ppl {r['ppl']:.1f}, "
                             f"{r['bpb']:.3f} bits/byte{wp} ({r['seconds']:.1f} s)")
        return r

    # -- probes ------------------------------------------------------------------------- #
    def paragraph_start(self, seed: int) -> int:
        """A validation offset just after a newline token (passages start at a line)."""
        arr = self.data["validation"]
        rng = np.random.default_rng(seed)
        n = len(arr)
        need = min(self.model_cfg.block_size, 256) + 2
        nl = np.array([b"\n" in self.data.tokenizer.token_bytes(i) for i in range(self.data.vocab_size)])
        for _ in range(50):
            o = int(rng.integers(0, max(1, n - need)))
            seg = np.asarray(arr[o:min(n, o + 400)])
            for h in np.nonzero(nl[seg])[0]:
                s = o + int(h) + 1
                if s + need <= n and not nl[int(arr[s])]:
                    return s
        return int(rng.integers(0, max(1, n - need)))

    def new_snippet(self, seed: int | None = None) -> None:
        seed = int(time.time() * 1000) if seed is None else seed
        self.snippet_start = self.paragraph_start(seed)
        self.snapshot = None

    def snippet_tokens(self) -> np.ndarray:
        s = self.snippet_start
        return np.asarray(self.data["validation"][s:s + self.snippet_len], dtype=np.int64)

    def run_probes(self) -> None:
        snap = snippet_analysis(self.model, self.snippet_tokens())
        snap["step"] = self.step
        self.snapshot = snap
        hs = head_scores(self.model, seed=self.cfg.seed)
        hs["step"] = self.step
        self.last_probe = hs
        if self.data.unigram is not None:
            self.embed = embedding_analysis(self.model, self.data.unigram, previous=self.embed) | {"step": self.step}
        rec = {"step": self.step, "tokens": self.tokens, "induction_max": float(hs["induction"].max()),
               "previous_max": float(hs["previous"].max()), "loss_first": hs["loss_first"],
               "loss_repeat": hs["loss_repeat"], "induction": hs["induction"].tolist(),
               "previous": hs["previous"].tolist()}
        self.hist["probe"].append(rec)
        if self.logger:
            self.logger.write("probes.csv", [self.step, self.tokens, rec["induction_max"], rec["previous_max"],
                                             rec["loss_first"], rec["loss_repeat"]])

    def generate(self, prompt: str, n: int = 60, temperature: float = 0.8, top_k: int = 0, top_p: float = 0.95,
                 seed: int = 0) -> str:
        ids = self.data.tokenizer.encode(prompt) or [0]
        s = Sampler(self.model, ids, temperature, top_k, top_p, seed)
        out = [s.step()["token"] for _ in range(n)]
        return self.data.tokenizer.decode(out)

    def run_sample(self) -> None:
        for i, prompt in enumerate(SAMPLE_PROMPTS):
            text = self.generate(prompt, 60, seed=self.cfg.seed + i)
            self.hist["samples"].append((self.step, prompt, text))
            if self.logger:
                self.logger.sample(self.step, prompt, text)
        del self.hist["samples"][:-40]

    # -- checkpoints ---------------------------------------------------------------------- #
    def state(self) -> dict:
        rng = {"torch": torch.get_rng_state()}
        if self.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state()
        elif self.device.type == "mps":
            rng["mps"] = torch.mps.get_rng_state()
        return {
            "format": CHECKPOINT_FORMAT,
            "model_config": self.model_cfg.to_dict(),
            "train_config": asdict(self.cfg),
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "step": self.step, "tokens": self.tokens, "train_seconds": self.train_seconds,
            "history": {k: v for k, v in self.hist.items()},
            "rng": rng,
            "tokenizer": self.data.tokenizer.to_dict(),
            "data": {"dataset": self.data.name, "fingerprint": self.data.fingerprint},
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "torch": torch.__version__,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        torch.save(self.state(), tmp)
        os.replace(tmp, path)

    def load_state(self, ck: dict) -> None:
        """Restore weights, optimizer, counters, history and RNG (same architecture).

        Slim checkpoints (no "optimizer" key, see paper_study/pack_results.py) load for
        evaluation and inspection; training on from them restarts the AdamW moments."""
        if ck.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("not a transformer_lab checkpoint")
        if GPTConfig(**ck["model_config"]) != self.model_cfg:
            raise ValueError("checkpoint architecture differs from this trainer's")
        if ck["data"]["fingerprint"] != self.data.fingerprint:
            raise ValueError(f"checkpoint was trained on different data/tokenizer ({ck['data']['dataset']})")
        self.model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            self.opt.load_state_dict(ck["optimizer"])
        else:
            self.add_log("warn", "slim checkpoint: no optimizer state; training on restarts the AdamW moments")
        self.step, self.tokens = int(ck["step"]), int(ck["tokens"])
        self.train_seconds = float(ck.get("train_seconds", 0.0))
        for k, v in ck.get("history", {}).items():
            self.hist[k] = list(v)
        rng = ck.get("rng", {})
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "cuda" in rng and self.device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"])
        if "mps" in rng and self.device.type == "mps":
            torch.mps.set_rng_state(rng["mps"])
        self.last_eval = None
        if self.hist["eval"]:
            e = dict(self.hist["eval"][-1])
            for k in ("pos_loss", "freq_loss", "calib_conf", "calib_acc", "calib_n"):
                if k in e:
                    e[k] = np.asarray(e[k], dtype=np.float64)
            self.last_eval = e
        self.last_probe = None
        self.snapshot = None
        self.stop_reason = None
        self.add_log("info", f"restored step {self.step:,} ({self.tokens / 1e6:.1f}M tokens, saved "
                             f"{ck.get('saved_at', '?')})")

    def close(self) -> None:
        if self.logger:
            self.logger.close()


def icl_score(pos_loss: np.ndarray) -> float:
    """Olsson et al. (2022): loss at the 500th token minus loss at the 50th (averaged over
    a few neighbouring positions); scaled to shorter contexts. Negative = later tokens are
    predicted better because the context is used."""
    T = len(pos_loss)
    if T < 20:
        return float("nan")
    late = (500, 510) if T >= 512 else (int(0.9 * T), int(0.9 * T) + max(1, T // 50))
    early = (45, 55) if T >= 512 else (int(0.09 * T), int(0.09 * T) + max(1, T // 50))
    a, b = np.nanmean(pos_loss[late[0]:late[1]]), np.nanmean(pos_loss[early[0]:early[1]])
    return float(a - b) if math.isfinite(a) and math.isfinite(b) else float("nan")


def load_checkpoint(path: str) -> dict:
    import pickle
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except (pickle.UnpicklingError, RuntimeError, EOFError, AttributeError) as e:
        raise ValueError(f"{path} is not a readable checkpoint ({type(e).__name__}: {e})") from e
    if not isinstance(ck, dict) or ck.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a transformer_lab checkpoint")
    return ck


def checkpoint_tokenizer(ck: dict) -> Tokenizer:
    return Tokenizer.from_dict(ck["tokenizer"])


def merge_train_config(saved: dict, overrides: dict) -> tuple[TrainConfig, list[str]]:
    """The saved training configuration with explicit command-line overrides applied."""
    names = {f.name for f in fields(TrainConfig)}
    base = {k: v for k, v in saved.items() if k in names}
    changed = []
    for k, v in overrides.items():
        if k in names and v is not None and base.get(k) != v:
            changed.append(f"{k}: {base.get(k)} -> {v}")
            base[k] = v
    return TrainConfig(**base), changed


# --------------------------------------------------------------------------- #
# Headless training
# --------------------------------------------------------------------------- #
def run_headless(trainer: Trainer, save_path: str | None, max_seconds: float | None = None,
                 max_tokens: int | None = None, status_every: float = 15.0, autosave_minutes: float = 10.0,
                 on_step=None) -> str:
    """Train until max_steps, a time or token budget, or Ctrl+C; save on the way out."""
    stop = {"flag": False}

    def on_sigint(signum, frame):
        if stop["flag"]:
            raise KeyboardInterrupt
        stop["flag"] = True
        print("\ninterrupt: finishing the current step and saving (Ctrl+C again to abort)", flush=True)

    old = signal.signal(signal.SIGINT, on_sigint)
    t_start = last_status = last_save = time.time()
    ema = None
    reason = "max_steps"
    printed_log = len(trainer.log)
    try:
        while True:
            if stop["flag"]:
                reason = "interrupted"
                break
            if max_seconds is not None and time.time() - t_start >= max_seconds:
                reason = "time budget"
                break
            if max_tokens is not None and trainer.tokens >= max_tokens:
                reason = "token budget"
                break
            st = trainer.advance()
            for kind, msg in trainer.log[printed_log:]:
                print(f"[{kind}] {msg}", flush=True)
            printed_log = len(trainer.log)
            if st is None:
                reason = trainer.stop_reason or "max_steps"
                break
            if on_step is not None:
                on_step(trainer)
            ema = st["loss"] if ema is None else 0.98 * ema + 0.02 * st["loss"]
            now = time.time()
            if now - last_status >= status_every:
                last_status = now
                left = (trainer.cfg.max_steps - trainer.step) * st["seconds"]
                print(f"step {trainer.step:,}/{trainer.cfg.max_steps:,}  tokens {trainer.tokens / 1e6:.1f}M  "
                      f"epoch {trainer.epoch:.3f}  loss {ema:.4f}  lr {st['lr']:.2e}  |g| {st['grad_norm']:.2f}  "
                      f"{st['tok_s']:,.0f} tok/s  ETA {left / 3600:.1f} h", flush=True)
            if save_path and autosave_minutes and now - last_save >= autosave_minutes * 60:
                trainer.save(save_path)
                last_save = now
                print(f"autosaved {save_path} at step {trainer.step:,}", flush=True)
    finally:
        signal.signal(signal.SIGINT, old)
        if save_path and trainer.step > 0:
            trainer.save(save_path)
            print(f"saved {save_path} (step {trainer.step:,}, {trainer.tokens / 1e6:.1f}M tokens); "
                  f"continue with --resume {save_path}", flush=True)
        if trainer.logger:
            trainer.logger.flush()
    return reason


def summary_line(trainer: Trainer) -> str:
    e = trainer.last_eval
    val = f"val loss {e['loss']:.4f} (ppl {e['ppl']:.1f}, {e['bpb']:.3f} bpb)" if e else "no evaluation yet"
    return (f"step {trainer.step:,}, {trainer.tokens / 1e6:.1f}M tokens ({trainer.epoch:.3f} epochs), "
            f"{trainer.train_seconds / 60:.1f} min training; {val}")


if __name__ == "__main__":
    print("run gpt_lab.py (dashboard / --headless) or evaluate_checkpoint.py", file=sys.stderr)
    sys.exit(2)
