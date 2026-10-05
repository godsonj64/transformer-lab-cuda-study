"""
model.py - a decoder-only transformer language model (GPT family), written out in full.

Architecture (defaults; every choice can be switched for ablations)
    token embedding  (V x d), no input scaling
    position         rotary embeddings on queries and keys (RoPE, Su et al. 2021), or
                     learned absolute position embeddings added to the input (GPT-2)
    N x block        pre-norm residual block:
                         x = x + W_o * Attention(Norm(x))      causal, multi-head
                         x = x + MLP(Norm(x))
                     Attention(h) = softmax(Q K^T / sqrt(d_head) + causal mask) V per head
                     MLP = W_down (SiLU(W_gate h) * W_up h)    (SwiGLU, Shazeer 2020)
                        or W_2 GELU(W_1 h)                     (GPT-2)
    final norm       RMSNorm (Zhang & Sennrich 2019) or LayerNorm
    unembedding      logits = Norm(x) W_E^T, tied to the token embedding (Press & Wolf 2017)

Initialisation follows GPT-2: N(0, 0.02) everywhere, residual output projections
N(0, 0.02 / sqrt(2 N)) so the residual stream's variance does not grow with depth.

The fast path uses torch's fused scaled_dot_product_attention. `capture=True` computes the
same attention explicitly and returns the attention probabilities of every head and the
residual stream after every block (for the attention maps and the logit lens); the tests
check that both paths give the same logits.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 16384
    block_size: int = 512            # context length in tokens
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    mlp: str = "swiglu"              # swiglu | gelu
    mlp_hidden: int = 0              # 0: 8/3 d rounded up to 64 (swiglu) or 4 d (gelu)
    norm: str = "rms"                # rms | layer
    pos: str = "rope"                # rope | learned
    rope_base: float = 10000.0
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
    norm_eps: float = 1e-5

    def validate(self) -> None:
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        if self.pos == "rope" and self.head_dim % 2:
            raise ValueError("RoPE needs an even head dimension")
        if self.mlp not in ("swiglu", "gelu") or self.norm not in ("rms", "layer") or self.pos not in ("rope", "learned"):
            raise ValueError("mlp must be swiglu|gelu, norm rms|layer, pos rope|learned")
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.d_model) < 1:
            raise ValueError("sizes must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def hidden(self) -> int:
        if self.mlp_hidden:
            return self.mlp_hidden
        if self.mlp == "swiglu":
            return int(math.ceil(8 * self.d_model / 3 / 64) * 64)
        return 4 * self.d_model

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * g  (computed in at least float32)."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float32 and self.weight.dtype == torch.float32:   # fused kernel, same formula
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        wide = torch.float64 if x.dtype == torch.float64 else torch.float32  # widen half precision only
        xf = x.to(wide)
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.to(wide)).type_as(x)


def make_norm(cfg: GPTConfig) -> nn.Module:
    if cfg.norm == "rms":
        return RMSNorm(cfg.d_model, cfg.norm_eps)
    return nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps, bias=cfg.bias)


def rope_tables(n_pos: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of the rotation angles pos * base^(-2i/d_head), shape (n_pos, d_head),
    each frequency repeated for the two halves of the head (GPT-NeoX pairing: dimension i
    rotates with dimension i + d_head/2)."""
    inv_freq = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    ang = torch.outer(torch.arange(n_pos, dtype=torch.float64), inv_freq)
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos().float(), ang.sin().float()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each (i, i + d/2) pair of x (..., T, d_head) by its position's angle. Then
    q_m . k_n depends on the positions only through m - n."""
    return x * cos.to(x.dtype) + rotate_half(x) * sin.to(x.dtype)


class KVCache:
    """Keys and values of the tokens seen so far, for incremental decoding."""

    def __init__(self, cfg: GPTConfig, batch: int, device, dtype=torch.float32):
        shape = (cfg.n_layer, batch, cfg.n_head, cfg.block_size, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0
        self.capacity = cfg.block_size

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t0, t1 = self.length, self.length + k.shape[2]
        if t1 > self.capacity:
            raise ValueError("KV cache is full: the context is limited to block_size tokens")
        self.k[layer, :, :, t0:t1] = k
        self.v[layer, :, :, t0:t1] = v
        return self.k[layer, :, :, :t1], self.v[layer, :, :, :t1]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer: int):
        super().__init__()
        self.cfg = cfg
        self.layer = layer
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.proj.is_residual_out = True

    def forward(self, x: torch.Tensor, rope, cache: KVCache | None = None, capture: dict | None = None) -> torch.Tensor:
        cfg = self.cfg
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, cfg.n_head, cfg.head_dim).permute(2, 0, 3, 1, 4)
        t0 = cache.length if cache is not None else 0
        if rope is not None:
            cos, sin = rope[0][t0:t0 + T], rope[1][t0:t0 + T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.append(self.layer, k, v)
        Tk = k.shape[2]
        drop = cfg.dropout if self.training else 0.0
        if capture is None:
            if Tk == T:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
            elif T == 1:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
            else:
                mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(diagonal=Tk - T)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(cfg.head_dim))
            mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(diagonal=Tk - T)
            att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
            capture["attn"].append(att)
            capture["q"].append(q.detach())
            capture["k"].append(k.detach())
            capture["v"].append(v.detach())
            if drop:
                att = F.dropout(att, drop)
            y = att @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        if capture is not None:
            capture["heads"].append(y.detach())          # head outputs, concatenated, before W_o
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.kind = cfg.mlp
        h = cfg.hidden
        if cfg.mlp == "swiglu":
            self.up = nn.Linear(cfg.d_model, 2 * h, bias=cfg.bias)      # [gate | up], fused
        else:
            self.up = nn.Linear(cfg.d_model, h, bias=cfg.bias)
        self.down = nn.Linear(h, cfg.d_model, bias=cfg.bias)
        self.down.is_residual_out = True

    def forward(self, x: torch.Tensor, capture: dict | None = None) -> torch.Tensor:
        if self.kind == "swiglu":
            gate, up = self.up(x).chunk(2, dim=-1)
            act = F.silu(gate) * up
            if capture is not None:
                capture["mlp_pre"].append(gate.detach())
                capture["mlp_up"].append(up.detach())
        else:
            pre = self.up(x)
            act = F.gelu(pre)
            if capture is not None:
                capture["mlp_pre"].append(pre.detach())
        if capture is not None:
            capture["mlp_act"].append(act.detach())
        return self.down(act)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, layer: int):
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg, layer)
        self.ln2 = make_norm(cfg)
        self.mlp = MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, rope, cache: KVCache | None = None, capture: dict | None = None) -> torch.Tensor:
        h1 = self.ln1(x)
        a = self.drop(self.attn(h1, rope, cache, capture))
        x = x + a
        h2 = self.ln2(x)
        m = self.drop(self.mlp(h2, capture))
        x = x + m
        if capture is not None:
            capture["ln1"].append(h1.detach())
            capture["ln2"].append(h2.detach())
            capture["attn_out"].append(a.detach())
            capture["mlp_out"].append(m.detach())
            capture["resid"].append(x.detach())
        return x


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
CAPTURE_KEYS = ("attn", "q", "k", "v", "heads", "attn_out", "ln1", "ln2", "mlp_pre", "mlp_up", "mlp_act",
                "mlp_out", "resid")


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model) if cfg.pos == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.ln_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        if cfg.pos == "rope":
            cos, sin = rope_tables(cfg.block_size, cfg.head_dim, cfg.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = self.rope_sin = None
        self.apply(self._init)
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "is_residual_out", False):
                nn.init.normal_(m.weight, 0.0, resid_std)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    # -- sizes ----------------------------------------------------------------- #
    def num_params(self, non_embedding: bool = False) -> int:
        """Distinct parameters (tied weights counted once). non_embedding excludes the token
        and position embeddings (the convention of Kaplan et al. 2020)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
            if self.wpe is not None:
                n -= self.wpe.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= 0                    # an untied unembedding is a real matmul: keep it
        return n

    def flops_per_token(self, seq_len: int | None = None) -> float:
        """Training FLOPs per token (forward + backward), PaLM appendix B:
        6 N + 12 L d T, with N the matmul parameters (unembedding included)."""
        cfg = self.cfg
        T = seq_len or cfg.block_size
        n_matmul = self.num_params(non_embedding=True) + cfg.d_model * cfg.vocab_size * (1 if cfg.tie_embeddings else 0)
        return 6.0 * n_matmul + 12.0 * cfg.n_layer * cfg.d_model * T

    def rope(self):
        return (self.rope_cos, self.rope_sin) if self.rope_cos is not None else None

    # -- forward --------------------------------------------------------------- #
    def embed(self, idx: torch.Tensor, t0: int = 0) -> torch.Tensor:
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(t0, t0 + idx.shape[1], device=idx.device))
        return self.drop(x)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                cache: KVCache | None = None, capture: bool = False, reduction: str = "mean"):
        """Returns (logits, loss) or, with capture=True, (logits, loss, internals) where
        internals has per layer: both norm outputs, queries / keys / values after RoPE
        (B, H, T, d_head), attention probabilities (B, H, T, T), the concatenated head
        outputs before W_o, the attention output, the MLP pre-activation(s) and hidden
        activation, the MLP output, and the residual stream after the embedding and after
        every block."""
        B, T = idx.shape
        t0 = cache.length if cache is not None else 0
        if t0 + T > self.cfg.block_size:
            raise ValueError(f"sequence of {t0 + T} tokens exceeds block_size {self.cfg.block_size}")
        cap = {k: [] for k in CAPTURE_KEYS} if capture else None
        x = self.embed(idx, t0)
        if cap is not None:
            cap["resid"].append(x.detach())
        rope = self.rope()
        for block in self.blocks:
            x = block(x, rope, cache, cap)
        if cache is not None:
            cache.length += T
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            lg = logits if logits.dtype in (torch.float32, torch.float64) else logits.float()   # widen bf16 only
            loss = F.cross_entropy(lg.view(-1, lg.size(-1)), targets.reshape(-1), reduction=reduction)
            if reduction == "none":
                loss = loss.view(B, T)
        if capture:
            return logits, loss, cap
        return logits, loss

    def lens_logits(self, resid: torch.Tensor) -> torch.Tensor:
        """Logit lens (nostalgebraist 2020): decode an intermediate residual stream with the
        final norm and the unembedding."""
        return self.lm_head(self.ln_f(resid))

    # -- optimizer -------------------------------------------------------------- #
    def param_groups(self, weight_decay: float) -> list[dict]:
        """Weight decay on matrices (linear layers and embeddings); none on norm gains and
        biases. Tied weights appear once."""
        decay, no_decay, seen = [], [], set()
        for _, p in self.named_parameters():
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            (decay if p.dim() >= 2 else no_decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]

    def tensor_groups(self) -> list[tuple[str, list[torch.nn.Parameter]]]:
        """Named weight matrices for the layer-health diagnostics."""
        out = [("embed", [self.wte.weight])]
        for i, b in enumerate(self.blocks):
            out.append((f"{i + 1}.qkv", [b.attn.qkv.weight]))
            out.append((f"{i + 1}.attn_out", [b.attn.proj.weight]))
            out.append((f"{i + 1}.mlp_in", [b.mlp.up.weight]))
            out.append((f"{i + 1}.mlp_out", [b.mlp.down.weight]))
        return out


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def filter_probs(logits: np.ndarray, temperature: float, top_k: int, top_p: float) -> np.ndarray:
    """The distribution a token is sampled from: logits / temperature, keep the top_k most
    likely tokens (0 = all), then the smallest set whose probability reaches top_p
    (nucleus sampling, Holtzman et al. 2020). temperature 0 = greedy."""
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        p = np.zeros_like(logits)
        p[int(np.argmax(logits))] = 1.0
        return p
    z = logits / temperature
    if 0 < top_k < len(z):
        kth = np.partition(z, -top_k)[-top_k]
        z = np.where(z >= kth, z, -np.inf)
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    if 0 < top_p < 1:
        order = np.argsort(-p, kind="stable")
        cum = np.cumsum(p[order])
        keep = order[:int(np.searchsorted(cum, top_p) + 1)]
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()
    return p


class Sampler:
    """Token-by-token generation with a KV cache. Every token is predicted from the last
    min(len, block_size) tokens: once the context is full, the window slides and the last
    block_size tokens are re-encoded for each new token (positions must restart at 0)."""

    def __init__(self, model: GPT, prompt: list[int], temperature: float = 0.8, top_k: int = 0,
                 top_p: float = 0.95, seed: int = 0):
        self.model = model
        self.tokens = list(prompt)
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.rng = np.random.default_rng(seed)
        self.device = next(model.parameters()).device
        self.cache: KVCache | None = None
        self.next_logits: np.ndarray | None = None
        if not self.tokens:
            raise ValueError("the prompt must contain at least one token")

    @torch.no_grad()
    def _prefill(self) -> None:
        ctx = self.tokens[-self.model.cfg.block_size:]
        self.cache = KVCache(self.model.cfg, 1, self.device, next(self.model.parameters()).dtype)
        logits, _ = self.model(torch.tensor([ctx], device=self.device), cache=self.cache)
        self.next_logits = logits[0, -1].float().cpu().numpy()

    @torch.no_grad()
    def step(self) -> dict:
        was_training = self.model.training
        self.model.eval()
        try:
            if self.next_logits is None:
                self._prefill()
            raw = self.next_logits
            p_raw = np.exp(raw - raw.max())
            p_raw /= p_raw.sum()
            p = filter_probs(raw, self.temperature, self.top_k, self.top_p)
            tok = int(self.rng.choice(len(p), p=p))
            top = np.argsort(-p_raw)[:12]
            nz = p_raw[p_raw > 0]
            info = {"token": tok, "p_model": float(p_raw[tok]), "p_sample": float(p[tok]),
                    "entropy": float(-(nz * np.log(nz)).sum()),
                    "top": [(int(i), float(p_raw[i]), float(p[i])) for i in top],
                    "context": min(len(self.tokens), self.model.cfg.block_size)}
            self.tokens.append(tok)
            if self.cache.length < self.model.cfg.block_size:
                logits, _ = self.model(torch.tensor([[tok]], device=self.device), cache=self.cache)
                self.next_logits = logits[0, -1].float().cpu().numpy()
            else:
                self._prefill()                       # sliding window
            return info
        finally:
            self.model.train(was_training)
