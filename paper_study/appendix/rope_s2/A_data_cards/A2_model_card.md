# Model card

**Model.** Decoder-only transformer language model (GPT family) trained from scratch: 6 layers,
6 heads, width 384, context 512 tokens, rope position encoding,
rms normalisation, swiglu MLP, tied input/output embeddings; 16,913,280 parameters
(10,621,824 non-embedding).

**Training.** 24,576,000 tokens (0.195 epochs) of wikitext-103, 3,000
updates of 16×1×512 tokens; AdamW (β = 0.9, 0.95,
weight decay 0.1 on matrices), peak learning rate 0.001, 500 warm-up updates,
cosine schedule over 20,000 updates, gradient clipping at 1.0; seed 2.
Training compute about 2.84e+15 FLOP.

**Evaluation.** Validation loss 4.0481 nats/token, perplexity 57.3, 1.3456 bits per byte, top-1 accuracy 31.64%, expected calibration error 0.0088, at update 3,000, over all 263,950 validation target tokens.

**Intended use.** Research and teaching about how transformer language models learn: training dynamics,
attention, interpretability measurements. Not for generating text that is relied upon.

**Limitations.** A small model trained briefly: generated text is often ungrammatical and factually wrong. It
reflects the content and biases of English Wikipedia's good and featured articles. Word-normalized perplexity is
computed on the raw text and is not the classic `<unk>`-based WikiText-103 benchmark number.
