# Appendix: transformer language model on wikitext-103

Exported 2026-10-05T12:24:31 at update 3,000 (24,576,000 training tokens). Figures: print themes; print figures are 183 mm wide (single-panel figures 89 mm) at 300 dpi, slides 1920 × 1080. Formats: png, pdf. Time-lapse frames: 500, 1,000, 1,500, 2,000, 2,500, 3,000.

## A. Data cards

**Figure A1. Run card.** Key numbers of the run at update 3,000: validation metrics over all validation tokens, training budget, model size and compute.

Files: [A1_run_card_print.png](A_data_cards\A1_run_card_print.png), [A1_run_card_print.pdf](A_data_cards\A1_run_card_print.pdf)

**Document A2. Model card.** Architecture, training, evaluation, intended use and limitations.

Files: [A2_model_card.md](A_data_cards\A2_model_card.md)

**Document A3. Data card.** Source, licence, composition, checksums and preprocessing of the corpus.

Files: [A3_data_card.md](A_data_cards\A3_data_card.md)

## B. Curves

**Figure B1. Training and validation loss.** Validation loss fell from 9.77 to 4.03 nats/token (perplexity 17,465 to 56.1) over 24.6M training tokens; dashed lines: unigram (7.29) and interpolated bigram (5.10) models fitted on the training split.

Files: [B1_loss_print.png](B_curves\B1_loss_print.png), [B1_loss_print.pdf](B_curves\B1_loss_print.pdf)

**Figure B2. Learning-rate schedule.** AdamW learning rate: linear warm-up over 500 updates, then cosine decay to 0.1× the peak 0.001 over 20,000 updates; dot: update 3,000.

Files: [B2_learning_rate_print.png](B_curves\B2_learning_rate_print.png), [B2_learning_rate_print.pdf](B_curves\B2_learning_rate_print.pdf)

**Figure B3. Gradient norm.** Global L2 norm of the gradient before clipping; clipped in 0% of the last 200 updates.

Files: [B3_gradient_norm_print.png](B_curves\B3_gradient_norm_print.png), [B3_gradient_norm_print.pdf](B_curves\B3_gradient_norm_print.pdf)

**Figure B4. Next-token accuracy.** Share of validation tokens in the model's top 1 / top 5: 31.9% / 52.3%.

Files: [B4_accuracy_print.png](B_curves\B4_accuracy_print.png), [B4_accuracy_print.pdf](B_curves\B4_accuracy_print.pdf)

**Figure B5. Calibration.** Reliability diagram at update 3,000 (15 confidence bins); expected calibration error 0.0109.

Files: [B5_calibration_print.png](B_curves\B5_calibration_print.png), [B5_calibration_print.pdf](B_curves\B5_calibration_print.pdf)

**Figure B7. Loss by token frequency.** Validation loss of target tokens grouped by their training-frequency rank.

Files: [B7_frequency_print.png](B_curves\B7_frequency_print.png), [B7_frequency_print.pdf](B_curves\B7_frequency_print.pdf)

**Figure B8. Induction heads over training.** Largest induction score per layer on repeated random sequences; latest maximum 0.558.

Files: [B8_induction_print.png](B_curves\B8_induction_print.png), [B8_induction_print.pdf](B_curves\B8_induction_print.pdf)

**Figure B9. In-context copying.** Loss on random token sequences and on their exact repeat; latest 12.43 vs 9.79 nats.

Files: [B9_copying_print.png](B_curves\B9_copying_print.png), [B9_copying_print.pdf](B_curves\B9_copying_print.pdf)

**Figure B6. Loss by context position.** Validation loss at each context position, averaged over windows, for 6 evaluations (colour: training tokens); in-context score (loss at token 500 minus token 50) -0.077.

Files: [B6_loss_by_position_print.png](B_curves\B6_loss_by_position_print.png), [B6_loss_by_position_print.pdf](B_curves\B6_loss_by_position_print.pdf)

**Figure B0. Training dynamics.** (a) validation loss fell from 9.77 to 4.03 nats/token (perplexity 17,465 to 56.1) over 24.6M training tokens; dashed lines: unigram (7.29) and interpolated bigram (5.10) models fitted on the training split; (b) AdamW learning rate: linear warm-up over 500 updates, then cosine decay to 0.1× the peak 0.001 over 20,000 updates; dot: update 3,000; (c) global L2 norm of the gradient before clipping; clipped in 0% of the last 200 updates.

Files: [B0_strip_training_dynamics_print.png](B_curves\B0_strip_training_dynamics_print.png), [B0_strip_training_dynamics_print.pdf](B_curves\B0_strip_training_dynamics_print.pdf)

**Figure B0. Evaluation.** (a) share of validation tokens in the model's top 1 / top 5: 31.9% / 52.3%; (b) reliability diagram at update 3,000 (15 confidence bins); expected calibration error 0.0109; (c) validation loss of target tokens grouped by their training-frequency rank.

Files: [B0_strip_evaluation_print.png](B_curves\B0_strip_evaluation_print.png), [B0_strip_evaluation_print.pdf](B_curves\B0_strip_evaluation_print.pdf)

**Figure B0. Mechanisms over training.** (a) largest induction score per layer on repeated random sequences; latest maximum 0.558; (b) loss on random token sequences and on their exact repeat; latest 12.43 vs 9.79 nats.

Files: [B0_strip_mechanisms_print.png](B_curves\B0_strip_mechanisms_print.png), [B0_strip_mechanisms_print.pdf](B_curves\B0_strip_mechanisms_print.pdf)

## C. Maps

**Figure C1. Attention of every head.** Attention probabilities softmax(QKᵀ/√d) of all 6 layers × 6 heads on the first 48 tokens of a validation passage at update 3,000 (rows: queries, columns: keys; square-root colour scale).

Files: [C1_attention_heads_step0003000_print.png](C_maps\C1_attention_heads_step0003000_print.png), [C1_attention_heads_step0003000_print.pdf](C_maps\C1_attention_heads_step0003000_print.pdf)

**Figure C2. Attention-head scores.** (a) induction, (b) previous-token and (c) duplicate-token scores of every head, measured on 48 random tokens repeated twice; (d) mean attention entropy on the validation passage; update 3,000.

Files: [C2_head_scores_step0003000_print.png](C_maps\C2_head_scores_step0003000_print.png), [C2_head_scores_step0003000_print.pdf](C_maps\C2_head_scores_step0003000_print.pdf)

**Figure C3. Logit lens.** Each layer's residual stream decoded with the final norm and unembedding (nostalgebraist 2020) for the first 16 positions of the passage; colour: probability of the actual next token; text: the layer's top prediction; update 3,000.

Files: [C3_logit_lens_step0003000_print.png](C_maps\C3_logit_lens_step0003000_print.png), [C3_logit_lens_step0003000_print.pdf](C_maps\C3_logit_lens_step0003000_print.pdf)

**Figure C4. Update size per weight matrix.** Relative update size ‖ΔW‖/‖W‖ of every weight matrix, measured every 10 updates over 300 measurements.

Files: [C4_layer_health_print.png](C_maps\C4_layer_health_print.png), [C4_layer_health_print.pdf](C_maps\C4_layer_health_print.pdf)

**Figure C5. Token embeddings.** All 16,384 token embeddings on the first two principal components of the frequency-weighted embedding distribution (31.8% and 4.8% of the variance); colour: training-frequency rank; labels: the most frequent tokens; update 3,000.

Files: [C5_embeddings_step0003000_print.png](C_maps\C5_embeddings_step0003000_print.png), [C5_embeddings_step0003000_print.pdf](C_maps\C5_embeddings_step0003000_print.pdf)

**Figure C6. Inside block 5.** Activations of block 5 for the first 32 tokens (rows) at update 3,000: (a) residual stream entering the block, (b) after RMSNorm, (c) attention of head 3 (the block's strongest induction head), (d) attention-weighted values of all heads, (e) the 96 most active of 1,024 MLP hidden units, (f) residual stream leaving the block; diverging panels use each panel's own symmetric scale.

Files: [C6_block5_internals_step0003000_print.png](C_maps\C6_block5_internals_step0003000_print.png), [C6_block5_internals_step0003000_print.pdf](C_maps\C6_block5_internals_step0003000_print.pdf)

**Figure C7. Layer 5, head 3 across training.** Attention pattern of layer 5, head 3 (the strongest induction head at the last capture) on the same validation passage at updates 500, 1,000, 1,500, 2,000, 2,500, 3,000; one colour scale for all panels.

Files: [C7_timelapse_attention_L5H3_print.png](C_maps\C7_timelapse_attention_L5H3_print.png), [C7_timelapse_attention_L5H3_print.pdf](C_maps\C7_timelapse_attention_L5H3_print.pdf)

**Figure C8. Induction scores across training.** Induction score of every head at updates 500, 1,000, 1,500, 2,000, 2,500, 3,000; shared colour scale 0–0.56.

Files: [C8_timelapse_induction_scores_print.png](C_maps\C8_timelapse_induction_scores_print.png), [C8_timelapse_induction_scores_print.pdf](C_maps\C8_timelapse_induction_scores_print.pdf)

**Figure C9. Logit lens across training.** Probability of the actual next token decoded from every layer for the first 16 positions of the same passage at updates 500, 1,000, 1,500, 2,000, 2,500, 3,000; shared colour scale.

Files: [C9_timelapse_logit_lens_print.png](C_maps\C9_timelapse_logit_lens_print.png), [C9_timelapse_logit_lens_print.pdf](C_maps\C9_timelapse_logit_lens_print.pdf)

**Figure C10. Token embeddings across training.** All token embeddings on each capture's first two frequency-weighted principal components at updates 500, 1,000, 1,500, 2,000, 2,500, 3,000; axis signs aligned between captures, shared limits; colour: training-frequency rank (blue rare, red frequent).

Files: [C10_timelapse_embeddings_print.png](C_maps\C10_timelapse_embeddings_print.png), [C10_timelapse_embeddings_print.pdf](C_maps\C10_timelapse_embeddings_print.pdf)

## D. Data sheets

**Table D1. Corpus statistics.** Rows, words (whitespace words plus one end-of-line per line, the WikiText convention), UTF-8 megabytes, BPE tokens and bytes per token of every split.

Files: [D1_corpus.csv](D_data_sheets\D1_corpus.csv), [D1_corpus.md](D_data_sheets\D1_corpus.md), [D1_corpus_print.png](D_data_sheets\D1_corpus_print.png), [D1_corpus_print.pdf](D_data_sheets\D1_corpus_print.pdf)

**Table D2. Reference models.** Validation cross-entropy of reference models fitted on the training split and of this transformer.

Files: [D2_reference_models.csv](D_data_sheets\D2_reference_models.csv), [D2_reference_models.md](D_data_sheets\D2_reference_models.md), [D2_reference_models_print.png](D_data_sheets\D2_reference_models_print.png), [D2_reference_models_print.pdf](D_data_sheets\D2_reference_models_print.pdf)

**Table D3. Validation metrics.** Every validation pass (35): loss, perplexity, bits per byte, word-normalized perplexity, top-1 / top-5 accuracy and expected calibration error, each over all validation target tokens.

Files: [D3_validation_metrics.csv](D_data_sheets\D3_validation_metrics.csv), [D3_validation_metrics.md](D_data_sheets\D3_validation_metrics.md), [D3_validation_metrics_print.png](D_data_sheets\D3_validation_metrics_print.png), [D3_validation_metrics_print.pdf](D_data_sheets\D3_validation_metrics_print.pdf)

**Table D4. Parameters.** Parameter counts by component (tied input/output embedding counted once); per-tensor shapes in the CSV.

Files: [D4_parameters.csv](D_data_sheets\D4_parameters.csv), [D4_parameters.md](D_data_sheets\D4_parameters.md), [D4_parameters_print.png](D_data_sheets\D4_parameters_print.png), [D4_parameters_print.pdf](D_data_sheets\D4_parameters_print.pdf), [D4_parameters_by_tensor.csv](D_data_sheets\D4_parameters_by_tensor.csv)

**Table D5. Head scores.** Every head's induction, previous-token and duplicate-token score and attention entropy at update 3,000, sorted by induction score (top 12 typeset; all in the CSV).

Files: [D5_head_scores.csv](D_data_sheets\D5_head_scores.csv), [D5_head_scores.md](D_data_sheets\D5_head_scores.md), [D5_head_scores_print.png](D_data_sheets\D5_head_scores_print.png), [D5_head_scores_print.pdf](D_data_sheets\D5_head_scores_print.pdf)

**Table D6. Hyperparameters.** Model architecture and training configuration.

Files: [D6_hyperparameters.csv](D_data_sheets\D6_hyperparameters.csv), [D6_hyperparameters.md](D_data_sheets\D6_hyperparameters.md), [D6_hyperparameters_print.png](D_data_sheets\D6_hyperparameters_print.png), [D6_hyperparameters_print.pdf](D_data_sheets\D6_hyperparameters_print.pdf)

**Table D7. Frames.** Capture steps used for the maps and time-lapse strips.

Files: [D7_frames.csv](D_data_sheets\D7_frames.csv), [D7_frames.md](D_data_sheets\D7_frames.md), [D7_frames_print.png](D_data_sheets\D7_frames_print.png), [D7_frames_print.pdf](D_data_sheets\D7_frames_print.pdf)

