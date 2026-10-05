"""
data.py - real text corpora, tokenized once into memory-mapped token files.

Corpora
    wikitext-103   Merity et al. (2016): 28,475 good and featured English Wikipedia articles,
                   103 million words of training text plus the standard validation and test
                   splits. The "raw" version (original casing, punctuation and numbers, no
                   <unk> substitution) from Hugging Face's Salesforce/wikitext repository.
    wikitext-2     The 2-million-word subset; same validation and test splits.
    <file.txt>     Any UTF-8 text file: the first 90% of its lines train, the next 5%
                   validate, the last 5% test.

Every downloaded file is checked against the SHA-256 published by the repository. The text
of a split is the concatenation of its dataset rows (each row is one line of the original
files; empty rows are blank lines).

Preparation (once per corpus and vocabulary size; results are cached in data/<corpus>/)
    1. count the regex pieces of the training split and learn byte-level BPE merges from
       them - the tokenizer never sees validation or test text;
    2. tokenize every split into a flat uint16 file (`<split>-<V>.bin`);
    3. record statistics: rows, words, bytes, tokens; and two reference models fitted on
       the training split - the unigram model (token frequencies only) and an interpolated
       bigram model - whose validation cross-entropy the transformer has to beat.

Word-level perplexity uses the WikiText convention: the number of words is the count of
whitespace-separated words plus one end-of-line per line (217,646 for validation and
245,569 for test, as in the WikiText paper).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from bpe import Tokenizer, count_pieces, train_bpe

HF_BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main"
_VALID = ("validation-00000-of-00001.parquet", 657209,
          "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c")
_TEST = ("test-00000-of-00001.parquet", 732610,
         "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91")
DATASETS = {
    "wikitext-103": {
        "config": "wikitext-103-raw-v1",
        "files": {
            "train": [("train-00000-of-00002.parquet", 156987808,
                       "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212"),
                      ("train-00001-of-00002.parquet", 157088770,
                       "ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720")],
            "validation": [_VALID], "test": [_TEST]},
    },
    "wikitext-2": {
        "config": "wikitext-2-raw-v1",
        "files": {
            "train": [("train-00000-of-00001.parquet", 6357543,
                       "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7")],
            "validation": [_VALID], "test": [_TEST]},
    },
}
SPLITS = ("train", "validation", "test")
META_FORMAT = "corpus_meta_v1"


def log_print(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Download and integrity
# --------------------------------------------------------------------------- #
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: str, size: int, sha256: str, log: Callable[[str], None] = log_print) -> None:
    """Fetch url to dest (via dest.part) and verify its size and SHA-256 before keeping it."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    log(f"downloading {url} ({size / 1e6:.1f} MB)")
    t0, last = time.time(), 0.0
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        got = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            got += len(block)
            if sys.stdout.isatty() and time.time() - last > 0.5:
                last = time.time()
                print(f"\r  {got / 1e6:7.1f} / {size / 1e6:.1f} MB", end="", flush=True)
    if sys.stdout.isatty():
        print()
    verify_file(part, size, sha256)
    os.replace(part, dest)
    log(f"  verified sha256 in {time.time() - t0:.0f} s")


def verify_file(path: str, size: int, sha256: str) -> None:
    actual = os.path.getsize(path)
    if actual != size:
        raise OSError(f"{path}: expected {size} bytes, found {actual}")
    digest = sha256_file(path)
    if digest != sha256:
        raise OSError(f"{path}: sha256 {digest} does not match the published {sha256}")


def ensure_raw(name: str, data_dir: str, log: Callable[[str], None] = log_print) -> dict[str, list[str]]:
    """Paths of the raw parquet files of every split, downloading and verifying missing ones."""
    spec = DATASETS[name]
    out: dict[str, list[str]] = {}
    for split, files in spec["files"].items():
        out[split] = []
        for fname, size, sha in files:
            path = os.path.join(data_dir, name, "raw", fname)
            if not os.path.exists(path):
                download(f"{HF_BASE}/{spec['config']}/{fname}", path, size, sha, log)
            elif os.path.getsize(path) != size:
                raise OSError(f"{path} has the wrong size; delete it to download it again")
            out[split].append(path)
    return out


# --------------------------------------------------------------------------- #
# Rows, text, word counts
# --------------------------------------------------------------------------- #
def read_parquet_rows(paths: list[str]) -> list[str]:
    import pyarrow.parquet as pq
    rows: list[str] = []
    for p in paths:
        rows.extend(pq.read_table(p, columns=["text"]).column("text").to_pylist())
    return rows


def normalize_rows(rows: list[str]) -> list[str]:
    """Every row as one line ending in a newline (empty rows are blank lines)."""
    return [r if r.endswith("\n") else r + "\n" for r in rows]


def standard_word_count(rows: list[str]) -> int:
    """WikiText convention: whitespace-separated words plus one end-of-line per line."""
    return sum(len(r.split()) for r in rows) + len(rows)


def text_chunks(rows: list[str], chunk_chars: int = 1 << 20) -> list[str]:
    """Join rows into chunks of about chunk_chars characters. A chunk only ends before a row
    that starts with visible text, where the pre-tokenization regex always splits, so
    tokenizing the chunks separately gives exactly the tokens of the whole text."""
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for r in rows:
        if size >= chunk_chars and r.strip():
            chunks.append("".join(cur))
            cur, size = [], 0
        cur.append(r)
        size += len(r)
    if cur:
        chunks.append("".join(cur))
    return chunks


@dataclass
class CorpusInfo:
    name: str
    dir: str
    vocab_size: int
    tokenizer: Tokenizer
    meta: dict

    def bin_path(self, split: str) -> str:
        return os.path.join(self.dir, f"{split}-{self.vocab_size}.bin")

    def split_stats(self, split: str) -> dict:
        return self.meta["splits"][split]

    @property
    def fingerprint(self) -> str:
        return self.meta["fingerprint"]


def load_rows(name: str, data_dir: str, log: Callable[[str], None] = log_print) -> dict[str, list[str]]:
    if name in DATASETS:
        paths = ensure_raw(name, data_dir, log)
        return {s: normalize_rows(read_parquet_rows(paths[s])) for s in SPLITS}
    if not os.path.isfile(name):
        raise ValueError(f"unknown dataset {name!r}: use {', '.join(DATASETS)} or a path to a UTF-8 text file")
    with open(name, encoding="utf-8") as f:
        lines = normalize_rows(f.read().splitlines(keepends=True))
    if len(lines) < 40:
        raise ValueError(f"{name}: need at least 40 lines to make train/validation/test splits")
    a, b = int(len(lines) * 0.90), int(len(lines) * 0.95)
    return {"train": lines[:a], "validation": lines[a:b], "test": lines[b:]}


def corpus_dir(name: str, data_dir: str) -> str:
    if name in DATASETS:
        return os.path.join(data_dir, name)
    with open(name, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(name))[0] or "text"
    return os.path.join(data_dir, f"{stem}-{digest}")


# --------------------------------------------------------------------------- #
# Reference models
# --------------------------------------------------------------------------- #
def unigram_baseline(train: np.ndarray, val: np.ndarray, vocab_size: int) -> dict:
    """Entropy of the training token distribution and the validation cross-entropy of the
    add-one-smoothed unigram model (nats per token)."""
    counts = np.bincount(train, minlength=vocab_size).astype(np.float64)
    p_train = counts / counts.sum()
    nz = p_train > 0
    entropy = float(-(p_train[nz] * np.log(p_train[nz])).sum())
    p = (counts + 1) / (counts.sum() + vocab_size)
    return {"train_entropy": entropy, "val_loss": float(-np.log(p[val]).mean()), "counts": counts}


def bigram_baseline(train: np.ndarray, val: np.ndarray, vocab_size: int,
                    max_tokens: int = 150_000_000) -> dict:
    """Interpolated bigram model P(b|a) = l * c(a,b)/c(a) + (1-l) * P_unigram(b), counted on
    the first 99% of the training tokens; l is chosen on the last 1% (held out); returns
    the validation cross-entropy in nats per token."""
    train = np.asarray(train[:max_tokens])
    cut = int(len(train) * 0.99)
    fit, held = train[:cut], train[cut:]
    uni = (np.bincount(fit, minlength=vocab_size) + 1.0) / (len(fit) + vocab_size)
    codes = fit[:-1].astype(np.int64) * vocab_size + fit[1:].astype(np.int64)
    keys, counts = np.unique(codes, return_counts=True)
    ctx = np.bincount(fit[:-1], minlength=vocab_size).astype(np.float64)

    def probs(seq: np.ndarray, lam: float) -> np.ndarray:
        a, b = seq[:-1].astype(np.int64), seq[1:].astype(np.int64)
        q = a * vocab_size + b
        i = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
        c_ab = np.where(keys[i] == q, counts[i], 0).astype(np.float64)
        c_a = ctx[a]
        p_bi = np.divide(c_ab, c_a, out=np.zeros_like(c_ab), where=c_a > 0)
        return lam * p_bi + (1 - lam) * uni[b]

    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    held_loss = [float(-np.log(probs(held, lam)).mean()) for lam in grid]
    best = float(grid[int(np.argmin(held_loss))])
    return {"lambda": best, "val_loss": float(-np.log(probs(np.asarray(val), best)).mean())}


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def prepare(name: str = "wikitext-103", data_dir: str = "data", vocab_size: int = 16384,
            workers: int | None = None, log: Callable[[str], None] = log_print) -> CorpusInfo:
    """Download (if needed), train the tokenizer, tokenize every split and compute the
    reference baselines. Everything is cached; a second call only reads the metadata."""
    if not 256 < vocab_size <= 65536:
        raise ValueError("vocab_size must be in 257..65536 (tokens are stored as uint16)")
    cdir = corpus_dir(name, data_dir)
    os.makedirs(cdir, exist_ok=True)
    meta_path = os.path.join(cdir, f"meta-{vocab_size}.json")
    tok_path = os.path.join(cdir, f"tokenizer-{vocab_size}.json")
    if os.path.exists(meta_path) and os.path.exists(tok_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        tok = Tokenizer.load(tok_path)
        info = CorpusInfo(name, cdir, vocab_size, tok, meta)
        if meta.get("format") == META_FORMAT and meta.get("tokenizer") == tok.fingerprint() and all(
                os.path.exists(info.bin_path(s)) and os.path.getsize(info.bin_path(s)) == 2 * meta["splits"][s]["tokens"]
                for s in SPLITS):
            return info
        log(f"{meta_path} is stale; preparing again")

    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    t_start = time.time()
    rows = load_rows(name, data_dir, log)
    stats = {s: {"rows": len(r), "words": standard_word_count(r)} for s, r in rows.items()}
    texts_bytes = {}
    for s, r in rows.items():
        n_bytes = sum(len(x.encode("utf-8")) for x in r)
        texts_bytes[s] = n_bytes
        stats[s]["bytes"] = n_bytes
        stats[s]["chars"] = sum(len(x) for x in r)
    log(f"{name}: " + ", ".join(f"{s} {stats[s]['words']:,} words / {stats[s]['bytes'] / 1e6:.1f} MB" for s in SPLITS))

    if os.path.exists(tok_path):
        tok = Tokenizer.load(tok_path)
        log(f"tokenizer: {tok_path}")
    else:
        chunks = text_chunks(rows["train"], 4 << 20)
        log(f"counting regex pieces of the training split ({len(chunks)} chunks, {workers} workers)")
        t0 = time.time()
        counts = count_pieces(chunks, workers=workers)
        log(f"  {len(counts):,} distinct pieces, {sum(counts.values()):,} in total ({time.time() - t0:.0f} s)")
        log(f"learning {vocab_size - 256:,} BPE merges")
        t0, last = time.time(), [0.0]

        def progress(n: int, total: int) -> None:
            if time.time() - last[0] > 5:
                last[0] = time.time()
                log(f"  {n:,} / {total:,} tokens ({time.time() - t0:.0f} s)")

        ranks = train_bpe(counts, vocab_size, progress)
        del counts
        if len(ranks) < vocab_size:
            raise ValueError(f"the training text only supports {len(ranks)} tokens; use a smaller --vocab-size")
        tok = Tokenizer(ranks, name=os.path.basename(cdir))
        tok.save(tok_path)
        log(f"  done in {time.time() - t0:.0f} s; saved {tok_path}")

    arrays: dict[str, np.ndarray] = {}
    for s in SPLITS:
        path = os.path.join(cdir, f"{s}-{vocab_size}.bin")
        t0 = time.time()
        chunks = text_chunks(rows[s], 1 << 20)
        tmp = path + ".tmp"
        n_tok = 0
        with open(tmp, "wb") as f:
            for i in range(0, len(chunks), 32):
                for ids in tok.encode_batch(chunks[i:i + 32]):
                    arr = np.asarray(ids, dtype=np.uint16)
                    arr.tofile(f)
                    n_tok += len(arr)
        os.replace(tmp, path)
        stats[s]["tokens"] = n_tok
        arrays[s] = np.memmap(path, dtype=np.uint16, mode="r")
        log(f"tokenized {s}: {n_tok:,} tokens, {texts_bytes[s] / max(n_tok, 1):.2f} bytes/token "
            f"({time.time() - t0:.0f} s)")
    del rows

    log("fitting unigram and bigram reference models on the training split")
    uni = unigram_baseline(arrays["train"], arrays["validation"], vocab_size)
    np.save(os.path.join(cdir, f"unigram-{vocab_size}.npy"), uni.pop("counts"))
    bi = bigram_baseline(arrays["train"], arrays["validation"], vocab_size)
    meta = {
        "format": META_FORMAT, "dataset": name, "vocab_size": vocab_size,
        "tokenizer": tok.fingerprint(), "splits": stats,
        "baselines": {"uniform": math.log(vocab_size), "unigram_entropy_train": uni["train_entropy"],
                      "unigram_val": uni["val_loss"], "bigram_val": bi["val_loss"],
                      "bigram_lambda": bi["lambda"]},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "prep_seconds": round(time.time() - t_start, 1),
    }
    if name in DATASETS:
        meta["source"] = {"repo": "Salesforce/wikitext", "config": DATASETS[name]["config"],
                          "files": {s: [[f, sha] for f, _, sha in fl] for s, fl in DATASETS[name]["files"].items()}}
    else:
        meta["source"] = {"file": os.path.abspath(name)}
    meta["fingerprint"] = hashlib.sha256(json.dumps(
        {"tokenizer": meta["tokenizer"], "tokens": {s: stats[s]["tokens"] for s in SPLITS},
         "dataset": name if name in DATASETS else meta["source"]}, sort_keys=True).encode()).hexdigest()
    with open(meta_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(meta_path + ".tmp", meta_path)
    log(f"baselines (validation nats/token): uniform {meta['baselines']['uniform']:.3f}, "
        f"unigram {uni['val_loss']:.3f}, bigram {bi['val_loss']:.3f}; prepared in {meta['prep_seconds']:.0f} s")
    return CorpusInfo(name, cdir, vocab_size, tok, meta)


# --------------------------------------------------------------------------- #
# Token access and batching
# --------------------------------------------------------------------------- #
class TokenData:
    """The memory-mapped token arrays of a prepared corpus (or in-memory arrays for tests)."""

    def __init__(self, info: CorpusInfo | None = None, arrays: dict[str, np.ndarray] | None = None,
                 tokenizer: Tokenizer | None = None, meta: dict | None = None):
        if info is not None:
            self.info = info
            self.tokenizer = info.tokenizer
            self.meta = info.meta
            self.arrays = {s: np.memmap(info.bin_path(s), dtype=np.uint16, mode="r") for s in SPLITS}
            path = os.path.join(info.dir, f"unigram-{info.vocab_size}.npy")
            self.unigram = np.load(path) if os.path.exists(path) else None
        else:
            if arrays is None or tokenizer is None:
                raise ValueError("TokenData needs a CorpusInfo or arrays + tokenizer")
            self.info = None
            self.tokenizer = tokenizer
            self.arrays = arrays
            self.meta = meta or arrays_meta(arrays, tokenizer)
            self.unigram = np.bincount(arrays["train"], minlength=tokenizer.vocab_size).astype(np.float64)
        self.vocab_size = self.tokenizer.vocab_size
        self.token_len = np.asarray(self.tokenizer.token_lengths(), dtype=np.int64)

    @property
    def name(self) -> str:
        return self.meta.get("dataset", "in-memory")

    @property
    def fingerprint(self) -> str:
        return self.meta["fingerprint"]

    def __getitem__(self, split: str) -> np.ndarray:
        return self.arrays[split]


def arrays_meta(arrays: dict[str, np.ndarray], tok: Tokenizer) -> dict:
    """Metadata for token arrays held in memory (the tests' corpora; the program itself always
    prepares a real corpus with prepare())."""
    stats = {}
    for s, a in arrays.items():
        text_bytes = int(np.asarray(tok.token_lengths())[np.asarray(a)].sum())
        stats[s] = {"rows": 0, "words": 0, "bytes": text_bytes, "chars": text_bytes, "tokens": int(len(a))}
    fp = hashlib.sha256(json.dumps({"tok": tok.fingerprint(), "n": {s: len(a) for s, a in arrays.items()}},
                                   sort_keys=True).encode()).hexdigest()
    return {"format": META_FORMAT, "dataset": "in-memory", "vocab_size": tok.vocab_size,
            "tokenizer": tok.fingerprint(), "splits": stats, "fingerprint": fp,
            "baselines": {"uniform": math.log(tok.vocab_size)}}


class BatchSampler:
    """Deterministic, resumable epochs over non-overlapping windows.

    The training tokens are cut into windows of block+1 tokens (inputs and shifted targets).
    Each epoch visits every window once in a random order, with the window grid shifted by a
    random offset so boundaries differ between epochs. Both depend only on (seed, epoch),
    so sample number g - and therefore the batch at any step - can be recomputed exactly
    after a restart, with no sampler state to save."""

    def __init__(self, n_tokens: int, block: int, seed: int):
        if n_tokens < 2 * (block + 1):
            raise ValueError(f"split has {n_tokens} tokens; needs at least {2 * (block + 1)} for block {block}")
        self.n_tokens, self.block, self.seed = n_tokens, block, seed
        self.n_windows = (n_tokens - 1 - block) // block
        self._cache: dict[int, tuple[np.ndarray, int]] = {}

    def _epoch(self, e: int) -> tuple[np.ndarray, int]:
        if e not in self._cache:
            if len(self._cache) > 2:
                self._cache.clear()
            rng = np.random.default_rng([self.seed, 0xE90C, e])
            self._cache[e] = (rng.permutation(self.n_windows), int(rng.integers(0, self.block)))
        return self._cache[e]

    def starts(self, first: int, count: int) -> np.ndarray:
        """Token offsets of samples first .. first+count-1."""
        g = np.arange(first, first + count, dtype=np.int64)
        out = np.empty(count, dtype=np.int64)
        for e in np.unique(g // self.n_windows):
            m = (g // self.n_windows) == e
            perm, shift = self._epoch(int(e))
            out[m] = shift + perm[g[m] % self.n_windows] * self.block
        return out

    def epoch_of(self, samples: int) -> float:
        return samples / self.n_windows


def gather(arr: np.ndarray, starts: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Inputs and next-token targets for windows starting at `starts`."""
    idx = starts[:, None] + np.arange(block + 1)[None, :]
    seq = np.asarray(arr[idx], dtype=np.int64)
    return np.ascontiguousarray(seq[:, :-1]), np.ascontiguousarray(seq[:, 1:])


def eval_windows(n_tokens: int, block: int, stride: int | None = None) -> list[tuple[int, int, int]]:
    """Sliding-window evaluation plan covering every target token of a split exactly once.

    Returns (start, length, first_scored) triples: the window feeds tokens
    start .. start+length-1 and scores the targets at window positions >= first_scored
    (predicting tokens start+first_scored+1 .. start+length). stride = block gives
    non-overlapping windows; a smaller stride gives every scored token at least
    block - stride tokens of context. Only the split's first token is never predicted
    (there is nothing to condition on)."""
    stride = block if stride is None else stride
    if not 1 <= stride <= block:
        raise ValueError("stride must be in 1..block")
    plan = []
    scored = 0                             # tokens 1..scored have been predicted
    for start in range(0, max(n_tokens - 1, 0), stride):
        end = min(start + block, n_tokens - 1)
        plan.append((start, end - start, scored - start))
        scored = end
        if end == n_tokens - 1:
            break
    return plan
