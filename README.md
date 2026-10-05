# Transformer Lab

The language-model counterpart of the maze and chess projects. It trains a GPT-style decoder-only transformer, written out in full in PyTorch, on **WikiText-103**: 103 million words of Wikipedia articles. A byte-level BPE tokenizer is learned from that corpus. A live pygame dashboard has ten tabs in the same visual style as the other two projects. Training can be paused, saved, resumed and continued, from the dashboard or headless.

Every chart, heat map, particle and number comes from the real model on the real corpus. Nothing is simulated or decorative. Randomness appears only where a method calls for it: token sampling in Generate, the data shuffle, the choice of validation passage, the bootstrap, and the random-token sequences used to measure induction heads.

![Training tab](docs/training.png)

| File | What it is |
|---|---|
| [gpt_lab.py](gpt_lab.py) | Entry point: dashboard or `--headless`, fresh or resumed runs, every option |
| [model.py](model.py) | The transformer: RoPE, RMSNorm, causal multi-head attention, SwiGLU MLP, tied embeddings, KV cache, sampling. Can capture every internal activation. |
| [bpe.py](bpe.py) | Byte-level BPE: GPT-4 split pattern, merge training, a reference encoder checked against tiktoken |
| [data.py](data.py) | Downloads and SHA-256 checks WikiText, tokenizes it once into memory-mapped files, fits unigram and bigram references, provides resumable batches and sliding-window evaluation plans |
| [train.py](train.py) | AdamW, LR schedules, gradient clipping and accumulation, evaluation metrics, atomic checkpoints, CSV logs, headless loop |
| [probes.py](probes.py) | Interpretability measurements: attention maps, logit lens, induction / previous-token / duplicate-token head scores, embedding PCA |
| [dashboard.py](dashboard.py) | The dashboard and the background training thread |
| [export.py](export.py) | Time-lapse capture and the editorial appendix: figures, strips, data sheets, model and data cards |
| [export_figures.py](export_figures.py) | Builds the appendix from a checkpoint and its captured frames |
| [evaluate_checkpoint.py](evaluate_checkpoint.py) | Full sliding-window evaluation of a frozen checkpoint on validation or test |
| [summarize_evaluations.py](summarize_evaluations.py) | Means, SDs and bootstrap intervals across independent training seeds |
| [research_guide.md](research_guide.md) | How to run controlled experiments and report them honestly |

## Quick launch

Using your existing environment (it already has torch, numpy, pygame-ce, tiktoken, regex and pyarrow). The data, checkpoint and logs are found next to `gpt_lab.py`, so it can be started from any folder:

```bash
/Users/godsonjohnson/Mazr/.venv/bin/python gpt_lab.py
```

The first run of a fresh setup downloads WikiText-103 (raw) from Hugging Face: about 315 MB, with every file checked against its published SHA-256. It then learns a 16,384-token BPE on the training split and tokenizes the corpus into `data/wikitext-103/`. On an Apple M4 the preparation took 63 s. It is already done in this folder.

`gpt_wikitext.pt` holds a **2-minute run** (100 updates, 0.8M tokens). Launching resumes it. To start from random weights instead:

```bash
/Users/godsonjohnson/Mazr/.venv/bin/python gpt_lab.py --fresh --overwrite
```

Train without a window for an hour, then watch the result:

```bash
/Users/godsonjohnson/Mazr/.venv/bin/python gpt_lab.py --headless --max-minutes 60
```

`--help` lists every option. `--device auto` (the default) uses CUDA, then Apple's GPU (MPS), then the CPU.

## Fresh isolated environment

```bash
python3 -m venv .venv
```

```bash
.venv/bin/python -m pip install -r requirements.txt
```

```bash
.venv/bin/python gpt_lab.py
```

## Pause, resume, continue

- **Pause:** `Space` in the dashboard. The training thread stops between updates, and every tab stays live.
- **Save:** `S`, plus an autosave every 10 minutes (`--autosave-minutes`). Closing the window, `Esc`, `Q` and Ctrl+C all save if anything trained since the last save. Headless Ctrl+C finishes the current update, then saves. Checkpoints are written to a temporary file and renamed, so an interrupted save never corrupts the previous one.
- **Resume:** run `gpt_lab.py` again. If the `--save` file exists it is resumed automatically; `--resume PATH` picks another file. A checkpoint holds the weights, the AdamW moments, the step and token counters, the configuration, the tokenizer, a fingerprint of the data and the **whole metric history**, so every chart continues where it stopped.
- **Continue past the planned end:** pass a larger `--max-steps` (default 20,000). The cosine schedule is redrawn over the new length, so the learning rate jumps at the resume step, and the program says so. For open-ended runs, start with `--schedule wsd` (warm-up, constant, short final decay).
- **Exactness:** batches are a pure function of (seed, step). A resumed run therefore sees exactly the batches it would have seen without the interruption. On the CPU, weights after an interruption are bit-for-bit identical (tested). On the M4 GPU, the run interrupted at step 74 reproduced the uninterrupted run's step-100 validation metrics in every logged digit (loss 6.8657, top-1 0.127357, ECE 0.0339861). MPS kernels are not guaranteed to be deterministic, so treat that as an observation, not a guarantee.
- Explicit flags override a resumed run's training settings (the program lists every change). Architecture flags are ignored on resume.

## Dashboard

Eleven tabs. Click them, press `1`–`9`, `0` and `-`, or `Tab` / `Shift+Tab`.

| Tab | What it shows |
|---|---|
| **Training** | Loss against tokens seen: training (per step and 50-step mean), validation on every target token, and the uniform / unigram / bigram references. Also the planned LR schedule with the current point, the gradient norm against the clip threshold, loss by context position (with Olsson et al.'s in-context score, loss at token 500 minus token 50), loss by token-frequency bucket, and per-tensor update size ‖ΔW‖/‖W‖ over time |
| **Data** | Splits with rows, words, bytes, tokens and bytes per token. Reference cross-entropies against this model. Zipf's law on the training tokens, with the fitted exponent. The batch the model just trained on, coloured by its per-token training loss. The first learned merges, the longest tokens, token lengths, epoch progress |
| **Architecture** | The whole model as a diagram with real parameter counts. Click a block to open it: the residual stream, both norms, Q/K/V of the selected head, the attention pattern of every head, the concatenated head outputs, the attention output, the MLP gate/up/hidden units/output and both residual additions, each a heat map of the real activations for the first 24 tokens of a validation passage. Below them is a table of the block's weight matrices with shapes, parameter counts, live RMS and update size |
| **Predict** | A 256-token validation passage coloured by surprisal −log p. Hover or click a token for the context, the actual next token with its probability and rank, and the model's top 8. Also a calibration (reliability) diagram with ECE, top-1/top-5 accuracy over training, and the rank histogram |
| **Attention** | Every head's attention map, the selected head in detail with token labels, and the passage shaded by the attention one token pays to earlier tokens |
| **Heads** | Induction, previous-token and duplicate-token scores of every head (measured on repeated random sequences), attention entropy, induction scores over training, loss on random tokens versus their repeat, and the strongest heads' patterns |
| **Logit Lens** | What each layer's residual stream would predict, decoded with the final norm and unembedding: top token, its probability, and the probability of the actual next token. Also lens loss by depth, residual-stream size, and how much each attention block and MLP writes |
| **Stream 3-D** | The residual stream as stacked planes (tokens × the 64 most variable channels, after the embedding and after each block). Links show the strongest attention between tokens. Input tokens sit at the bottom, the model's guesses at the top |
| **Embeddings** | All 16,384 token embeddings in 3-D on the top three frequency-weighted principal components (the PCA of a token's embedding as tokens occur in the text), drawn as glowing additive particles. Colour runs blue → red with training-frequency rank; brightness is log frequency × depth. Click a token for cosine nearest neighbours. Also explained variance, embedding norm against frequency, and attention by distance per layer, with the RoPE wavelengths |
| **Export** | Time-lapse capture (every N updates, within a chosen window, with or without PNGs of every dashboard panel) and a one-click appendix export with theme, width, format and section choices, progress and a preview of the exported figures |
| **Generate** | Type a prompt and sample with temperature, top-k and top-p. Tokens are coloured by the model's probability. Each step lists the candidates (model probability and sampling probability) and the entropy. Training pauses while this tab is open |

The right column always shows run statistics (speed, TFLOP/s, ETA, checkpoint state), six live graphs and the log.

![Architecture tab](docs/architecture.png)

| Keys | Action |
|---|---|
| `Space` / `N` | Pause and resume / one update while paused |
| `E` | Evaluate on the whole validation split now |
| `R` | New validation passage for every probe view |
| `S` / `L` | Save / load the checkpoint |
| `1`–`9`, `0`, `Tab` | Views |
| `↑↓` `←→` | Layer and head (Architecture, Attention, Heads, Stream 3-D); scroll (Logit Lens); move the pin (Predict) |
| `A` | Mean of heads (Attention) |
| `X` | Log / linear token axis (Training) |
| drag, wheel, `O` | Orbit, zoom, auto-orbit (3-D views) |
| `C` / `Shift+C` | Full screenshot / click a panel to capture it (`Esc` cancels) |
| `F11` | Full window and back |
| `Esc` / `Q` | Quit and save (in Generate, `Esc` returns to the previous view) |

In **Generate**: type the prompt, `Enter` generates, `Shift+Enter` adds a newline, `↑↓` temperature, `Shift+↑↓` top-k, `←→` top-p, `PgUp/PgDn` length, or use the `−`/`+` buttons.

Screenshots and logs go to `runs/<timestamp>/` (`--log-dir`, `--no-log`): `config.json`, `metrics.csv` (every update), `eval.csv`, `probes.csv` and `samples.txt`.

## Exporting figures for manuscripts and slides

**Capture.** In the Export tab, set an interval (every 10–2,500 updates), an optional end, and whether to save dashboard panels, then press *start recording*; *capture now* (`K`) takes a single frame. Each frame stores the measurements at that update: attention maps, logit lens, head scores, all token embeddings, the internals of every block, and the latest evaluation. With panels on, it also stores a PNG of every panel of every tab (about 5 MB per frame), drawn while training waits for a moment. Frames accumulate in `exports/frames-<checkpoint>/` across sessions. Headless runs capture too:

```bash
/Users/godsonjohnson/Mazr/.venv/bin/python gpt_lab.py --headless --max-minutes 60 --record-every 250 --record-panels --export-on-exit
```

**Export.** Press *export appendix* (`Enter`), or run `export_figures.py`. It writes `exports/appendix-<checkpoint>-step<N>-<time>/`, laid out like a manuscript appendix:

| Folder | Contents |
|---|---|
| `README.md`, `manifest.json` | An index of every figure and table, with a legend written from the measured numbers. Provenance: checkpoint, step, tokens, data fingerprint, frames, options |
| `A_data_cards/` | Run card (key numbers), model card and data card (source, licence, composition, checksums, preprocessing) |
| `B_curves/` | Loss, learning rate, gradient norm, accuracy, calibration, loss by position and frequency, induction heads, copying. Related curves are combined into lettered strips (`B0_strip_*`) |
| `C_maps/` | Every attention head, head-score maps, logit lens, update sizes, token embeddings and the internals of the block with the strongest induction head. Also **time-lapse strips** of the same map across captured updates, on one colour scale |
| `D_data_sheets/` | Corpus, reference models, validation metrics, parameters, head scores, hyperparameters and frames: CSV, Markdown and typeset table figures |
| `E_dashboard_frames/` | A time-lapse strip of every captured dashboard panel |

**Editorial format.**
- The *print* theme has a white background, Arial, and 7 pt text at exact journal widths: 183 mm double column, 120 mm, or 89 mm (single-panel figures are always 89 mm). It is saved as 300-dpi PNG, PDF with embedded TrueType fonts, and SVG with editable text.
- The *slides* theme is the dashboard's black at exactly 1920 × 1080.
- The palettes are colour-blind-safe (Okabe–Ito lines, viridis / magma / cividis maps).
- Every token and character renders, including CJK, through a font fallback chain.
- A full export of both themes in all three formats takes about 20–25 s and 25 MB.

## The model and its training

| | Default | Alternatives |
|---|---|---|
| Layers × heads × width | 6 × 6 × 384 (head dimension 64), context 512 | `--n-layer --n-head --d-model --block-size` |
| Positions | rotary embeddings on Q and K (RoPE, base 10,000) | `--pos learned` (GPT-2) |
| Norm | pre-norm RMSNorm | `--norm layer` |
| MLP | SwiGLU, 1,024 hidden units | `--mlp gelu` (4 d) |
| Embeddings | tied input/output, no bias | |
| Parameters | 16.91M (10.62M non-embedding) | |
| Tokenizer | byte-level BPE, 16,384 tokens, GPT-4 split pattern, trained on the training split only | `--vocab-size` |
| Batch | 16 × 512 tokens per update | `--batch-size --grad-accum` |
| Optimiser | AdamW β = (0.9, 0.95), weight decay 0.1 on matrices only, global norm clip 1.0 | |
| Schedule | linear warm-up 500, cosine to 0.1× over 20,000 updates (164M tokens ≈ 1.3 epochs) | `--schedule wsd/constant` |
| Initialisation | N(0, 0.02); residual output projections N(0, 0.02/√(2·layers)) | |

The 16k vocabulary is a measured choice. On the M4 GPU, the 50,257-token GPT-2 vocabulary spent about 1 s of a 1.3 s step in the output layer. 16,384 tokens trained about 2× faster, at 4.29 bytes per token on the training text.

Evaluation scores every target token of the validation split, so nothing is sampled. It reports loss (nats/token), perplexity, **bits per byte** (independent of the tokenizer, so comparable across vocabularies), word-level perplexity (WikiText convention: whitespace words plus one end-of-line per line, which reproduces the published 217,646 validation and 245,569 test words), top-1/top-5 accuracy, expected calibration error, loss by position and loss by frequency. During training, validation uses non-overlapping 512-token windows (about 12 s). `evaluate_checkpoint.py` uses sliding windows (default stride 256), so every token has at least 256 tokens of context.

The reference models are fitted on the training split. **Unigram**: 7.287 nats/token on validation. **Interpolated bigram**: 5.096, with λ chosen on held-out training text. Uniform guessing is 9.704.

## What to expect

Measured on an Apple M4 (10 cores, 16 GB, MPS):

- Training runs at about 9,300–10,400 tokens/s, 1.1–1.2 TFLOP/s of model FLOPs. That is about 0.87 s per update, and 20,000 updates take roughly 5 hours plus evaluation. The dashboard holds 30 frames/s while training at full speed in its own thread.
- The included **2-minute run** (100 updates, 0.82M tokens): validation loss fell from 9.77 to 6.87 (perplexity 959, 2.282 bits/byte, top-1 12.7%). That is better than the unigram model and not yet as good as the bigram. A 5-minute pilot reached 5.75 after 300 updates. Samples at that stage are WikiText-flavoured word salad.
- These short runs check the pipeline. They say nothing about the quality reachable with hours of training. Measure that with [research_guide.md](research_guide.md).
- Induction heads (Heads tab) typically appear abruptly after a loss plateau. Expect the induction score to stay near zero in a run of a few minutes.

![Embeddings](docs/embeddings.png)

![Logit lens](docs/logit_lens.png)

## Tests

```bash
/Users/godsonjohnson/Mazr/.venv/bin/python -m pytest -q
```

The suite has 104 tests and runs in about 45 seconds without the corpus download (the real-corpus test skips when it is missing):

- **Mathematics.** Fused and explicit attention give the same logits. Changing a future token never changes an earlier prediction. RoPE is a rotation whose dot products depend only on relative position. RMSNorm matches its formula. KV-cache decoding equals the full forward pass, including after the context window slides. Autograd gradients match finite differences in float64. The captured Q/K/V, head outputs, MLP activations and residuals rebuild every block exactly. The logit lens of the last layer equals the output. Parameter counts match the closed form. The sampling filters behave as specified.
- **Tokenizer.** Merge order and tie-breaking. Exact round trips for any Unicode. The reference encoder equals tiktoken's on the learned vocabulary (and on cl100k_base when it is cached). Tokenizing in chunks equals tokenizing the whole text.
- **Data.** Each epoch visits every window exactly once in a seeded order. Sliding-window plans score every target once with the promised context. Text-file corpora prepare end to end, and decoding returns the exact text. Downloads with the wrong size or hash are rejected. The reference models are exact on a predictable sequence. On the real corpus, the word counts are 103,227,021 / 217,646 / 245,569.
- **Training.** Pausing and resuming is bit-exact on the CPU (weights, AdamW moments and history). Checkpoints refuse other architectures or other data. Each metric is internally consistent (perplexity, bits per byte, word perplexity). Training lowers validation loss below the unigram reference. The head-score probe gives exact scores on known attention patterns. The headless loop honours its budget and saves.
- **Dashboard.** Every tab, the keys, Generate (typing, sampling, training held), the Architecture block and head selection, screenshots, a 500-event random-input fuzz test, save on close, and the threaded worker pausing and stopping cleanly, and the particle renderer placing coloured light at each point. A layout audit checks that no label overlaps another or leaves the canvas in any 2-D tab, at both a tiny shape and the shipped 6 × 6 shape.
- **Export.** The recorder's schedule, frames that round-trip, an appendix whose files all exist and are indexed, exact 183 mm / 89 mm widths at 300 dpi and 1920 × 1080 slides, embedded and editable fonts in PDF and SVG, every token label rendering in the print fonts, the Export tab recording panels and exporting, and headless `--record-every --record-panels --export-on-exit`.
- **CLI and research tools.** Fresh start, automatic resume, continuing with a larger `--max-steps`, refusing to overwrite without `--overwrite`, evaluation and seed summaries.

## Problems found and fixed while building it

- **Training on its own targets (MPS).** In the first pilot, training loss fell to 0.2 within 200 updates while validation loss *rose*. With torch 2.12.1, a `non_blocking=True` copy of a strided CPU array to the Apple GPU delivered stale memory instead of the batch. Batches are now contiguous and copied with blocking. A regression test compares the GPU batch with the CPU batch.
- **Hidden float32 in a float64 model.** The finite-difference gradient test found that the loss and RMSNorm narrowed float64 inputs to float32. They now only widen half-precision inputs. Float32 training is unaffected.
- **Generation context.** After the 512-token window filled, the sampler alternated between 512 and 511 tokens of context. It now always re-encodes the last 512 tokens. A test checks it against naive greedy decoding.
- **float64 on the Apple GPU.** The full-vocabulary PCA first cast the embedding to float64 on the device, which MPS cannot do. It now copies to the CPU first.
- **Arrow keys crashed the dashboard.** A drawing helper had the same name as the arrow-key handler. The random-input fuzz test found it.
- **The induction-head test taught a lesson.** Repeating fixed-length random segments trained a *positional* copy head (attend 12 tokens back), not an induction head. Variable lengths removed the shortcut, but the model then sat on the known loss plateau for longer than a unit test can wait. The probe is therefore tested on exactly known attention patterns. See the research guide.

These checks cover the exercised cases. They do not prove the absence of bugs.

## Disk

`data/wikitext-103/` takes 542 MB (315 MB raw parquet files plus 252 MB of training tokens). A checkpoint takes about 195 MB: weights plus the two AdamW moment buffers. Only one is kept per `--save` path.

The repository includes the six paper-study runs as **slim checkpoints** in `paper_study/ckpt/` (about 70 MB each): the float32 weights, configuration, tokenizer and full metric history, without the AdamW moments. They load in `evaluate_checkpoint.py`, `export_figures.py`, `paper_study/analyze.py`, `paper_study/final_probes.py` and the dashboard, but cannot continue a headless run. To retrain the study from scratch, delete `paper_study/ckpt/` and `paper_study/done_*` first.
