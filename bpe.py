"""
bpe.py - byte-level byte-pair-encoding tokenizer, trained from scratch on the corpus.

This is the tokenizer family used by GPT-2/3/4: text is split into pieces by a regular
expression (the GPT-4 "cl100k" pattern: words with their leading space, runs of at most
three digits, punctuation runs, whitespace), every piece is turned into UTF-8 bytes, and
learned merges join frequent adjacent byte sequences into single tokens. Every byte has
its own token, so any string round-trips exactly and nothing is ever "unknown".

    train_bpe(piece_counts, vocab_size) -> ranks       learn the merges
    Tokenizer(ranks).encode(text) / .decode(ids)       use them

A tokenizer is stored as `ranks`: token bytes -> token id, where ids 0-255 are the single
bytes and id 256 + i is the i-th merge. That is tiktoken's format, so encoding runs in
tiktoken's Rust core when it is installed; `bpe_encode_piece` is the reference
implementation of the same algorithm (merge the adjacent pair whose concatenation has the
lowest rank, leftmost first, until no pair is in the vocabulary), and the tests check that
both give identical tokens.
"""
from __future__ import annotations

import base64
import hashlib
import heapq
import json
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable

import regex

# The pre-tokenization pattern of OpenAI's cl100k_base (GPT-4) encoding.
GPT4_PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
                r"""|\s*[\r\n]|\s+(?!\S)|\s+""")
TOKENIZER_FORMAT = "byte_bpe_v1"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _count_chunk(args: tuple[str, str]) -> Counter:
    text, pattern = args
    return Counter(regex.findall(pattern, text))


def count_pieces(chunks: Iterable[str], pattern: str = GPT4_PATTERN, workers: int = 1,
                 progress: Callable[[int], None] | None = None) -> Counter:
    """Frequency of every pre-tokenized piece (as UTF-8 bytes) over an iterable of text
    chunks. Chunks are split independently, so they should end at line boundaries."""
    total: Counter = Counter()
    jobs = ((c, pattern) for c in chunks)
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            for i, part in enumerate(pool.imap_unordered(_count_chunk, jobs, chunksize=1)):
                total.update(part)
                if progress:
                    progress(i + 1)
    else:
        for i, job in enumerate(jobs):
            total.update(_count_chunk(job))
            if progress:
                progress(i + 1)
    out: Counter = Counter()
    for piece, n in total.items():
        out[piece.encode("utf-8")] += n
    return out


def train_bpe(piece_counts: dict[bytes, int], vocab_size: int,
              progress: Callable[[int, int], None] | None = None) -> dict[bytes, int]:
    """Learn byte-level BPE merges from piece frequencies.

    Each step merges the most frequent adjacent pair of tokens, counted over all pieces
    weighted by frequency (ties: the lexicographically smallest pair of byte strings), and
    gives the concatenation the next id. Pair counts are updated incrementally: only the
    pieces that contain the merged pair are touched. If a pair's concatenation is already a
    token (reachable by another split), the occurrences are merged into that token and no
    id is spent, so ids and byte strings stay one-to-one."""
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256 (one token per byte)")
    vocab: list[bytes] = [bytes([i]) for i in range(256)]
    ranks: dict[bytes, int] = {b: i for i, b in enumerate(vocab)}
    words: list[list[int]] = []
    freq: list[int] = []
    for piece, n in piece_counts.items():
        if n > 0 and len(piece) > 0:
            words.append(list(piece))
            freq.append(int(n))

    pair_count: dict[tuple[int, int], int] = defaultdict(int)
    where: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, w in enumerate(words):
        f = freq[wi]
        for p in zip(w, w[1:]):
            pair_count[p] += f
            where[p].add(wi)
    heap = [(-c, vocab[a], vocab[b], a, b) for (a, b), c in pair_count.items()]
    heapq.heapify(heap)

    while len(vocab) < vocab_size and heap:
        negc, _, _, a, b = heapq.heappop(heap)
        c = pair_count.get((a, b), 0)
        if c <= 0 or -negc != c:
            continue                      # stale entry: a fresher one was pushed when the count changed
        merged = vocab[a] + vocab[b]
        new = ranks.get(merged)
        if new is None:
            new = len(vocab)
            vocab.append(merged)
            ranks[merged] = new
        changed: set[tuple[int, int]] = set()
        for wi in where.pop((a, b), ()):
            w = words[wi]
            nw: list[int] = []
            i, n, hit = 0, len(w), False
            while i < n:
                if i + 1 < n and w[i] == a and w[i + 1] == b:
                    nw.append(new)
                    i += 2
                    hit = True
                else:
                    nw.append(w[i])
                    i += 1
            if not hit:
                continue                  # the pair left this piece through an earlier merge
            f = freq[wi]
            for p in zip(w, w[1:]):
                pair_count[p] -= f
                changed.add(p)
            for p in zip(nw, nw[1:]):
                pair_count[p] += f
                where[p].add(wi)
                changed.add(p)
            words[wi] = nw
        pair_count.pop((a, b), None)
        for p in changed:
            cnt = pair_count.get(p, 0)
            if cnt > 0:
                heapq.heappush(heap, (-cnt, vocab[p[0]], vocab[p[1]], p[0], p[1]))
            elif p in pair_count:
                del pair_count[p]
        if progress:
            progress(len(vocab), vocab_size)
    return ranks


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def bpe_encode_piece(piece: bytes, ranks: dict[bytes, int]) -> list[int]:
    """Reference BPE for one piece: repeatedly merge the adjacent pair whose concatenation
    has the lowest rank (leftmost on ties) until no adjacent pair is a token."""
    parts = [piece[i:i + 1] for i in range(len(piece))]
    while len(parts) > 1:
        best, best_rank = -1, None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best_rank is None or r < best_rank):
                best, best_rank = i, r
        if best < 0:
            break
        parts[best:best + 2] = [parts[best] + parts[best + 1]]
    return [ranks[p] for p in parts]


class Tokenizer:
    """Byte-level BPE tokenizer. Uses tiktoken's Rust encoder when available and the
    pure-Python reference otherwise (identical output, much slower)."""

    def __init__(self, ranks: dict[bytes, int], pattern: str = GPT4_PATTERN, name: str = "bpe"):
        ids = sorted(ranks.values())
        if ids != list(range(len(ids))):
            raise ValueError("token ids must be 0..n-1")
        if any(bytes([i]) not in ranks for i in range(256)):
            raise ValueError("every single byte needs a token")
        self.ranks = dict(ranks)
        self.pattern = pattern
        self.name = name
        self.vocab_size = len(ranks)
        self.decoder: list[bytes] = [b""] * self.vocab_size
        for b, i in ranks.items():
            self.decoder[i] = b
        self._re = regex.compile(pattern)
        self._cache: dict[bytes, list[int]] = {}
        try:
            import tiktoken
            self._tk = tiktoken.Encoding(name=f"{name}-{self.fingerprint()[:12]}", pat_str=pattern,
                                         mergeable_ranks=self.ranks, special_tokens={})
        except Exception:                                  # tiktoken missing: reference encoder
            self._tk = None

    # -- encode / decode ------------------------------------------------------ #
    def encode_reference(self, text: str) -> list[int]:
        out: list[int] = []
        for piece in self._re.findall(text):
            b = piece.encode("utf-8")
            ids = self._cache.get(b)
            if ids is None:
                ids = bpe_encode_piece(b, self.ranks)
                if len(self._cache) < 500_000:
                    self._cache[b] = ids
            out.extend(ids)
        return out

    def encode(self, text: str) -> list[int]:
        if self._tk is not None:
            return self._tk.encode_ordinary(text)
        return self.encode_reference(text)

    def encode_batch(self, texts: list[str], threads: int = 8) -> list[list[int]]:
        if self._tk is not None:
            return self._tk.encode_ordinary_batch(texts, num_threads=threads)
        return [self.encode_reference(t) for t in texts]

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(self.decoder[int(i)] for i in ids)

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_bytes(self, i: int) -> bytes:
        return self.decoder[int(i)]

    def token_lengths(self) -> list[int]:
        """UTF-8 byte length of every token (for bits-per-byte)."""
        return [len(b) for b in self.decoder]

    def display(self, i: int) -> str:
        """A visible label for one token: leading/trailing spaces as '·', newlines as '⏎',
        and bytes that are not complete UTF-8 characters as <xx>."""
        b = self.decoder[int(i)]
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            return "".join(f"<{x:02x}>" for x in b)
        s = s.replace("\n", "⏎").replace("\t", "\\t").replace("\r", "\\r")
        s = "".join(ch if ch.isprintable() or ch == " " else f"<{ord(ch):02x}>" for ch in s)   # control bytes, e.g. NUL
        if s.startswith(" "):
            s = "·" + s[1:]
        if s.endswith(" ") and len(s) > 1:
            s = s[:-1] + "·"
        return s if s.strip() else s.replace(" ", "·")

    # -- persistence ---------------------------------------------------------- #
    def to_dict(self) -> dict:
        tokens = [base64.b64encode(b).decode("ascii") for b in self.decoder]
        return {"format": TOKENIZER_FORMAT, "name": self.name, "pattern": self.pattern, "tokens": tokens}

    @classmethod
    def from_dict(cls, d: dict) -> Tokenizer:
        if d.get("format") != TOKENIZER_FORMAT:
            raise ValueError(f"not a {TOKENIZER_FORMAT} tokenizer")
        ranks = {base64.b64decode(t): i for i, t in enumerate(d["tokens"])}
        if len(ranks) != len(d["tokens"]):
            raise ValueError("duplicate tokens in tokenizer file")
        return cls(ranks, d["pattern"], d.get("name", "bpe"))

    def fingerprint(self) -> str:
        h = hashlib.sha256(self.pattern.encode("utf-8"))
        for b in self.decoder:
            h.update(len(b).to_bytes(2, "little"))
            h.update(b)
        return h.hexdigest()

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> Tokenizer:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
