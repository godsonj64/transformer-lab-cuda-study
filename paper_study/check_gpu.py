"""Check that PyTorch can train on this GPU; with --bench, time 20 training updates of the
study model (6 x 6 x 384, context 512, batch 16, float32) on random tokens."""
from __future__ import annotations

import os
import sys
import time

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print(f"torch {torch.__version__}: CUDA not available")
        return 1
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    archs = torch.cuda.get_arch_list()
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(0)} ({arch}), "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    if arch not in archs:
        print(f"this build has no kernels for {arch}: {archs}")
        return 1
    x = torch.randn(256, 256, device="cuda")
    float((x @ x).sum())                         # fails here if the kernels do not run
    if "--bench" not in sys.argv:
        return 0
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import GPT, GPTConfig
    torch.manual_seed(0)
    m = GPT(GPTConfig()).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    idx = torch.randint(0, 16384, (16, 513), device="cuda")
    for i in range(25):
        if i == 5:
            torch.cuda.synchronize()
            t0 = time.time()
        _, loss = m(idx[:, :-1], idx[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 20
    print(f"benchmark: {dt:.3f} s/update, {16 * 512 / dt:,.0f} tokens/s (float32), "
          f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB. "
          f"Estimated 3,000 updates: {3000 * dt / 60:.0f} min per run plus evaluation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
