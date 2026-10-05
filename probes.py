"""
probes.py - measurements of what the network computes, from exact forward passes.

    snippet_analysis   one validation passage: per-token loss and top predictions, the
                       attention probabilities of every head, the logit lens (what each
                       layer's residual stream would predict), residual-stream norms
    head_scores        induction and previous-token scores of every head on repeated
                       random token sequences (Olsson et al. 2022, "In-context learning and
                       induction heads"), and the loss on the first vs the repeated half
    embedding_analysis principal components of the token embeddings of the most frequent
                       tokens; cosine nearest neighbours
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT


def _eval_mode(model: GPT):
    class _Ctx:
        def __enter__(self_inner):
            self_inner.was = model.training
            model.eval()

        def __exit__(self_inner, *exc):
            model.train(self_inner.was)
    return _Ctx()


@torch.no_grad()
def snippet_analysis(model: GPT, tokens: np.ndarray, k: int = 8) -> dict:
    """Everything the dashboard shows about one passage of len(tokens) tokens: the model
    reads tokens[:-1] and predicts tokens[1:]."""
    with _eval_mode(model):
        dev = next(model.parameters()).device
        tokens = np.asarray(tokens, dtype=np.int64)
        x = torch.tensor(tokens[None, :-1], device=dev)
        y = torch.tensor(tokens[1:], device=dev)
        T = x.shape[1]
        logits, _, cap = model(x, capture=True)
        logp = logits[0].float().log_softmax(-1)
        ar = torch.arange(T, device=dev)
        lp_true = logp[ar, y]
        top_lp, top_id = logp.topk(k, dim=-1)
        rank = (logp > lp_true[:, None]).sum(-1)
        p = logp.exp()
        entropy = -(p * logp).sum(-1)

        attn = torch.stack([a[0] for a in cap["attn"]])                    # (L, H, T, T)
        ent_h = -(attn * attn.clamp_min(1e-12).log()).sum(-1).mean(-1)    # (L, H)
        L = attn.shape[0]
        max_d = min(T, 64)
        dist = torch.zeros(L, max_d, device=dev)
        for d in range(max_d):
            q = torch.arange(d, T, device=dev)
            dist[:, d] = attn[:, :, q, q - d].mean(dim=(1, 2))

        lens_top, lens_p_top, lens_p_true, lens_nll = [], [], [], []
        resid_rms = []
        for r in cap["resid"]:
            lg = model.lens_logits(r[0]).float().log_softmax(-1)
            tp, ti = lg.max(-1)
            lens_top.append(ti)
            lens_p_top.append(tp.exp())
            lens_p_true.append(lg[ar, y].exp())
            lens_nll.append(-lg[ar, y])
            resid_rms.append(r[0].float().pow(2).mean(-1).sqrt())
        resid = torch.stack([r[0] for r in cap["resid"]]).float()          # (L+1, T, d)

        def npy(t, dtype=np.float32):
            return t.detach().float().cpu().numpy().astype(dtype)

        return {
            "tokens": tokens,
            "nll": npy(-lp_true),
            "rank": rank.cpu().numpy(),
            "entropy": npy(entropy),
            "top_id": top_id.cpu().numpy(),
            "top_p": npy(top_lp.exp()),
            "attn": npy(attn, np.float16),
            "head_entropy": npy(ent_h),
            "attn_by_distance": npy(dist),
            "lens_top": torch.stack(lens_top).cpu().numpy(),
            "lens_p_top": npy(torch.stack(lens_p_top)),
            "lens_p_true": npy(torch.stack(lens_p_true)),
            "lens_nll": npy(torch.stack(lens_nll)),
            "resid_rms": npy(torch.stack(resid_rms)),
            "resid": npy(resid, np.float16),
            "block": block_internals(cap, min(T, 32)),
            "attn_out_rms": npy(torch.stack([a[0].float().pow(2).mean(-1).sqrt() for a in cap["attn_out"]])),
            "mlp_out_rms": npy(torch.stack([m[0].float().pow(2).mean(-1).sqrt() for m in cap["mlp_out"]])),
        }


def block_internals(cap: dict, n: int) -> dict:
    """Every intermediate of every block for the first n tokens (float16), for the
    architecture view: norms, Q/K/V, head outputs, attention and MLP outputs, MLP hidden."""
    out = {}
    for key in ("ln1", "heads", "attn_out", "ln2", "mlp_pre", "mlp_up", "mlp_act", "mlp_out"):
        if cap.get(key):
            out[key] = torch.stack([t[0, :n] for t in cap[key]]).float().cpu().numpy().astype(np.float16)
    for key in ("q", "k", "v"):
        out[key] = torch.stack([t[0, :, :n] for t in cap[key]]).float().cpu().numpy().astype(np.float16)
    return out


@torch.no_grad()
def head_scores(model: GPT, seq_len: int = 48, batch: int = 8, seed: int = 0,
                token_range: tuple[int, int] | None = None) -> dict:
    """Score every head on sequences of seq_len random tokens repeated twice.

    induction  mean attention from position i of the repeat to position i - seq_len + 1:
               the token that followed the current token last time (the induction-head
               pattern [A][B] ... [A] -> [B])
    previous   mean attention from position i to i - 1
    duplicate  mean attention from position i of the repeat to i - seq_len (the earlier
               copy of the current token)
    The loss on the repeated half against the first half measures in-context copying:
    the first half is unpredictable random tokens."""
    with _eval_mode(model):
        dev = next(model.parameters()).device
        S = min(seq_len, model.cfg.block_size // 2)
        V = model.cfg.vocab_size
        rng = np.random.default_rng(seed)
        low, high = token_range or ((256 if V > 512 else 0), V)              # default: skip raw byte tokens
        r = rng.integers(low, high, size=(batch, S))
        x = torch.tensor(np.concatenate([r, r], axis=1), device=dev)
        logits, _, cap = model(x, capture=True)
        att = torch.stack(cap["attn"])                                      # (L, B, H, 2S, 2S)
        q = torch.arange(S, 2 * S, device=dev)
        induction = att[:, :, :, q, q - S + 1].mean(dim=(1, 3))
        duplicate = att[:, :, :, q, q - S].mean(dim=(1, 3))
        qa = torch.arange(1, 2 * S, device=dev)
        previous = att[:, :, :, qa, qa - 1].mean(dim=(1, 3))
        nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, V), x[:, 1:].reshape(-1),
                              reduction="none").view(batch, -1)
        first = float(nll[:, :S - 1].mean())
        second = float(nll[:, S - 1:].mean())
        return {
            "induction": induction.float().cpu().numpy(),
            "previous": previous.float().cpu().numpy(),
            "duplicate": duplicate.float().cpu().numpy(),
            "loss_first": first, "loss_repeat": second, "seq_len": S,
            "example_tokens": x[0].cpu().numpy(),
            "example_attn": att[:, 0].float().cpu().numpy().astype(np.float16),   # (L, H, 2S, 2S)
        }


def pca_coords(X: np.ndarray, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2
    return Xc @ vt[:n].T, var[:n] / max(var.sum(), 1e-30)


def align_signs(coords: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    """Principal axes have arbitrary signs; flip each to agree with the previous snapshot
    so the point cloud does not jump between frames."""
    if previous is None or previous.shape != coords.shape:
        return coords
    out = coords.copy()
    for j in range(coords.shape[1]):
        if float((coords[:, j] * previous[:, j]).sum()) < 0:
            out[:, j] *= -1
    return out


@torch.no_grad()
def embedding_analysis(model: GPT, counts: np.ndarray, previous: dict | None = None) -> dict:
    """Every token embedding projected on the top three principal components of the
    frequency-weighted embedding distribution (the PCA of the embedding of a token drawn
    from the training text). Returned in order of training frequency (rank 1 first)."""
    E = model.wte.weight.detach().cpu().double().numpy()      # float64 on the CPU (MPS has no float64)
    V = len(E)
    c = np.zeros(V)
    c[:min(V, len(counts))] = np.asarray(counts, dtype=np.float64)[:V]
    w = (c + 1.0) / (c + 1.0).sum()
    mu = w @ E
    X = (E - mu) * np.sqrt(w)[:, None]
    evals, evecs = np.linalg.eigh(X.T @ X)                      # d x d weighted covariance
    evals, evecs = evals[::-1], evecs[:, ::-1]
    ids = np.argsort(-c, kind="stable")
    coords = (E[ids] - mu) @ evecs[:, :3]
    if previous is not None and np.array_equal(previous.get("ids"), ids):
        coords = align_signs(coords, previous.get("coords"))
    out = {"ids": ids, "coords": coords.astype(np.float32),
           "explained": (evals[:3] / max(evals.sum(), 1e-30)).astype(np.float32),
           "norms": np.linalg.norm(E[ids], axis=1).astype(np.float32), "counts": c[ids]}
    if model.wpe is not None:
        P = model.wpe.weight.detach().float().cpu().numpy()
        Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        step = max(1, len(P) // 128)
        out["pos_sim"] = (Pn[::step] @ Pn[::step].T).astype(np.float32)
    return out


@torch.no_grad()
def nearest_neighbours(model: GPT, token: int, k: int = 12) -> list[tuple[int, float]]:
    E = model.wte.weight.detach().float()
    En = E / E.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sims = En @ En[int(token)]
    sims[int(token)] = -math.inf
    val, idx = sims.topk(k)
    return [(int(i), float(v)) for i, v in zip(idx.cpu(), val.cpu())]


def rope_wavelengths(head_dim: int, base: float) -> np.ndarray:
    """Wavelength in tokens of every rotary frequency pair: 2 pi / base^(-2i/d)."""
    i = np.arange(0, head_dim, 2, dtype=np.float64)
    return 2 * math.pi * base ** (i / head_dim)
