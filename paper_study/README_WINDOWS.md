# Running the paper study on a Windows laptop with an NVIDIA GPU (CUDA)

This folder is the Transformer Lab code plus the already-tokenized WikiText-103 data. It runs the same experiment as the Mac:

- 6-layer GPT-style model, 16.9M parameters
- RoPE vs learned absolute positions
- 3,000 updates per run
- float32 precision

**What runs here:** the full study, 6 runs (seeds 0, 1, 2 × RoPE / learned). All results in the paper come from this GPU, so every run shares the same hardware and precision. The Mac's seed-0 pair serves only as a cross-hardware check.

## 1. One-time setup (≈10–20 min, mostly downloads)

1. Plug the laptop into power. In *Settings → System → Power*:
   - set the power mode to *Best performance*;
   - set *Put my device to sleep* to *Never* (while plugged in) for the duration of the runs.
2. Update the NVIDIA driver (GeForce Experience / NVIDIA App, or nvidia.com). RTX 50-series GPUs need a recent driver.
3. Install **Python 3.12 (64-bit)** from <https://www.python.org/downloads/windows/>. In the installer, tick **"Add python.exe to PATH"**.
4. Unzip this archive to a short path without spaces, e.g. `C:\tlab\`. Long paths inside OneDrive folders can break.
5. Double-click **`paper_study\setup_windows.bat`**. It:
   - creates a virtual environment in `.venv`;
   - installs a PyTorch CUDA build that supports your GPU (it tries CUDA 13.0, 12.9 and 12.8; RTX 50-series needs 12.8 or newer);
   - installs the other packages;
   - runs a short benchmark that prints the time per update and an estimate per run.

## 2. Run the study

Double-click **`paper_study\run_windows.bat`**. Each run trains, saves its checkpoint, exports its figure appendix, and is evaluated once on validation and test. Progress lines show the update, the loss, tokens/s and the ETA.

- **Stop:** Ctrl+C in the window. The current update finishes, then it saves.
- **Resume:** double-click `run_windows.bat` again. Finished runs are skipped and an interrupted run resumes from its last save.
- **Some seeds only:** open a terminal in the folder and run, e.g., `paper_study\run_windows.bat --seeds 0 1`.

When all runs finish, the script writes **`paper_study\results_<computer>_<time>.zip`**, about 70 MB per run. It holds:

- slimmed checkpoints (weights plus the full metric history; no optimizer state);
- logs and evaluations;
- the exported figures.

Copy that zip to the Mac (USB drive, AirDrop from a phone, cloud drive…), put it in `Mazr2/transformer_project/paper_study/` for analysis.

## Do not change these

- Keep `--dtype float32` (the default here), so that the CUDA runs can be compared with the Mac's float32 seed-0 pair.
- Do not edit the training flags in `run_study.py`. They are the same as the Mac's `run_queue.sh`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Python was not found` | Install Python 3.12 with "Add python.exe to PATH", open a new window, run setup again |
| `no kernels for sm_120` / `CUDA not available` | Update the NVIDIA driver, then run setup again. Check `nvidia-smi` in a terminal |
| `CUDA out of memory` | Close games, browsers with video, and other GPU programs. The model needs about 2–3 GB |
| Very slow (under ~10,000 tokens/s) | Plug in AC power, set *Best performance*, and in the NVIDIA Control Panel set Python to use the high-performance NVIDIA GPU |
| `pip` fails for one package | Setup retries with the newest versions automatically. If it still fails, save the full error text and check the package versions in `requirements-windows.txt` |
| Window closes immediately | Open *Command Prompt*, `cd C:\tlab`, run `paper_study\setup_windows.bat` (or `run_windows.bat`) to see the message |

Files: `paper_study\run_study.py` (runner), `paper_study\check_gpu.py` (GPU check and benchmark), `paper_study\pack_results.py` (results zip), `gpt_lab.py` (training program), `README.md` (the Transformer Lab manual).
