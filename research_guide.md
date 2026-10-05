# Research, analysis and reporting guide

This is an experimental protocol for `gpt_lab.py`. It does not claim that the model reaches any particular perplexity.

## Define the question

A suitable question is: **do rotary position embeddings beat learned absolute positions at the same token budget?** Compare the default run with `--pos learned`, keeping every other setting fixed. Other single-factor questions include SwiGLU against GELU (`--mlp gelu`), RMSNorm against LayerNorm, the peak learning rate, or the context length. Changing two settings at once answers neither question.

Some changes alter the parameter count, for example `--mlp gelu`, which uses 4·d hidden units against SwiGLU's 8/3·d with three matrices. Report the counts, and decide in advance whether you compare at equal tokens, equal parameters or equal compute (`flops_per_token` in the dashboard). Different choices can give different answers.

Keep the splits apart. **Train** is for training. **Validation** is for every decision: settings, learning rate, when to stop. **Test** is evaluated once, at the end, for the runs you report. The tokenizer is trained on the training split only.

## Run a small installation check

From the project folder:

```bash
research_python="/Users/godsonjohnson/Mazr/.venv/bin/python"
study_dir="research_runs"

"$research_python" gpt_lab.py --headless --fresh --overwrite --max-tokens 2000000 --seed 0 \
  --save "$study_dir/pilot.pt" --log-dir "$study_dir/pilot"
```

This checks execution and logging only. Run pilots on validation before choosing a budget.

## Train independent replicates

Use at least 3–5 training seeds per condition, a fixed token budget, and otherwise identical flags. The seed sets the initial weights and the data order. Batches are a pure function of (seed, step), so seed *s* sees the same batches in every condition with the same batch size and context.

```bash
for condition_name in rope learned; do
  study_flags=()
  case "$condition_name" in
    learned) study_flags=(--pos learned) ;;
  esac
  for training_seed in 0 1 2; do
    "$research_python" gpt_lab.py --headless --fresh --overwrite --max-steps 6000 --max-tokens 49152000 \
      --seed "$training_seed" "${study_flags[@]}" \
      --save "$study_dir/train/${condition_name}_${training_seed}.pt" \
      --log-dir "$study_dir/train/${condition_name}_${training_seed}"
  done
done
```

`--max-steps` sets the length of the learning-rate schedule. Set it to the budget (here 6,000 updates × 8,192 tokens = 49.2M tokens) so every run finishes its cosine decay. A run cut off mid-schedule is compared at a high learning rate. Report tokens, updates, wall-clock time and the device.

Runs interrupted and resumed with unchanged settings continue exactly, so pausing does not contaminate an experiment. Changing `--batch-size`, `--grad-accum`, `--seed` or the schedule on resume does change the run; the program prints every such change.

## Evaluate frozen checkpoints

```bash
for condition_name in rope learned; do
  for training_seed in 0 1 2; do
    "$research_python" evaluate_checkpoint.py --checkpoint "$study_dir/train/${condition_name}_${training_seed}.pt" \
      --split validation --stride 256 --out-dir "$study_dir/eval/${condition_name}_${training_seed}"
  done
done
```

The evaluator never trains. It scores every validation token once: windows of 512 tokens, each after the first scoring only its last 256, so every scored token has at least 256 tokens of context. Use the same stride for every run you compare; the summarizer refuses to mix strides.

## Summarize across training seeds

```bash
"$research_python" summarize_evaluations.py \
  --condition rope "$study_dir"/eval/rope_* \
  --condition learned "$study_dir"/eval/learned_* \
  --split validation --out-dir "$study_dir/report"
```

The bootstrap resamples training seeds, the units that independently replicate the training process. It never resamples tokens: the ~264k validation tokens of one run are not independent replicates. With three seeds the interval is rough. Report the per-seed values (`per_run.csv`) next to the means.

Only after all choices are fixed, evaluate the reported runs once on test (`--split test`).

## Report

State:

- the data: WikiText-103 raw, the split, the tokenizer (byte-level BPE, 16,384 tokens, trained on train), tokens per byte;
- the model: layers, heads, width, context, position scheme, norm, MLP, parameter counts;
- the training: tokens, updates, batch, learning rate and schedule, weight decay, clipping, seeds, device, time;
- the evaluation: split, stride, number of scored tokens;
- the results: loss, **bits per byte**, perplexity, per seed and summarized.

**Comparing with published WikiText-103 numbers needs care.** Token-level perplexity depends on the tokenizer: a 16k-token perplexity is not comparable with a 50k-token or word-level one. Bits per byte is comparable across tokenizers on the same text. The word-level perplexity here follows the WikiText word count, but it is computed on the *raw* text, where rare words must be spelled out. The classic word-level benchmark replaces words outside a 267k vocabulary with `<unk>`, which is easier. The text is also the Hugging Face rows joined as lines, with empty rows as blank lines, and the first token of a split is never predicted. Call it "word-normalized perplexity on WikiText-103 raw", not the classic benchmark number.

## Interpretability measurements

The Heads tab and `probes.head_scores` measure attention on repeated random token sequences, where the second copy can be predicted only by copying from the first. The induction score of a head is its mean attention from each position of the repeat to the token that followed the earlier copy (Olsson et al., 2022). These are behavioural measurements on one input distribution. A high score shows the attention pattern, not by itself the whole circuit; ablate the head and measure the loss to establish its role.

A lesson from building the tests: training a small model on random segments of a **fixed** length taught it to attend a fixed distance back. That is a positional shortcut that also scores well on fixed-offset probes. With variable lengths the shortcut disappears, and the model sits on the known plateau before induction heads form. Probes should vary whatever a shortcut could exploit.

The logit lens decodes intermediate residual streams with the final norm and unembedding. Early layers are not trained to be decoded this way, so lens predictions there are a rough view, not the layer's "belief".

## Reproducibility

- On the CPU, a run is bit-for-bit reproducible from its seed. Pausing and resuming does not change it (tested).
- On MPS and CUDA, kernels are not guaranteed to be deterministic. Seeds then reproduce the data order and initial weights, not every bit. One resumed MPS run matched its uninterrupted twin to every logged digit, but do not rely on that.
- `torch==2.12.1` on MPS has a non-blocking-copy problem that once made the model train on its own targets. The code avoids it. If you change the data path, keep the regression test passing.
- Keep each run's `config.json`, the checkpoint's SHA-256 (`evaluate_checkpoint.py` records it) and `data/wikitext-103/meta-16384.json` (sizes, hashes and the data fingerprint).
