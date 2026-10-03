# Fine-tuning CodeBERT on Colab

Everything the paper's CodeBERT experiment (RQ6) could not do on CPU: **fine-tune** `microsoft/codebert-base` instead of using frozen embeddings.
The script reproduces exactly the same 30 000-function subsample, scenarios, seeds and splits as the CPU pipeline (verified: identical function hashes and part sizes), so its scores slot straight into the existing analysis.

## Files
| file | purpose |
|---|---|
| `CodeBERT_finetune.ipynb` | the notebook to open in Colab (GPU runtime) |
| `finetune_codebert.py` | resumable fine-tuning/scoring script (`--prepare`, `--dummy`, `--smoke`, normal run) |
| `universe_checksum.json` | checksums of the subsample; `--prepare` stops if the downloaded data differ |
| `bundle.zip` | the 5 files needed on Colab (fallback if the repo is private) |
| `../src/cb_splits.py` | shared, deterministic reconstruction of universe and splits |
| `../src/merge_ft.py` | merges Colab scores into `results/cbn_cache` (aligned on function hash, asserted) |

## Steps
1. Open `CodeBERT_finetune.ipynb` in Colab, set **Runtime → GPU**, run the cells top to bottom (configuration → Drive → code → prepare → dry run → run).
2. Seed 0 over all seven scenarios first (≈1.5–2.5 h on a T4, *estimate*), then optionally seeds 1 and 2. Results are written to Drive; re-running the cell resumes after a disconnect.
3. Download `ft_codebert_scores.zip`, unzip as `ft_out/` in the repository and run:
   ```
   python src/merge_ft.py ft_out
   python src/analyze_cb.py norm
   python src/make_cb.py          # tables/figure gain a "CodeBERT (fine-tuned)" column
   ```
   The paper then needs the new numbers written into Section 5.6 (RQ6) and the "frozen" caveats in Sections 1, 4 and 7 updated.

## What is trained
* Input: **normalised text** (comments removed, whitespace collapsed), first 256 tokens (`--max-len`). Do not use raw text: Big-Vul's formatting artifact (paper, Table "Formatting audit") would be learned.
* Model: `RobertaForSequenceClassification` (2 classes), AdamW, lr 2e-5, weight decay 0.01, batch 16, 3 epochs, 10 % linear warm-up then linear decay, gradient clipping 1.0, mixed precision (bf16 if supported, else fp16).
* Training data: the scenario's training part of the subsample (≈5–13 k functions, 33 % vulnerable). **Early stopping** uses a 10 % validation split carved from the training part, never the calibration data, so the conformal guarantees remain valid.
* Score: `logit(vulnerable) − logit(benign)`. The triage layer only uses the ranking, as for all other scorers.
* Output rows are in the same order as `results/cbn_cache/*.parquet`; `merge_ft.py` asserts equal hashes and labels.

## Checks that were run locally (CPU)
* `--dummy` end-to-end (splits, I/O, resume, merge, analysis) on random scores.
* `--smoke` real training/evaluation of CodeBERT on 48 functions (loss decreases, parquet written).
* The reconstructed universe and the splits of five scenario/seed pairs equal the cached CPU run exactly.
* **Not run:** a full GPU training. Expect to look at the first epochs' validation AUROC and adjust `--lr`/`--epochs` only on the validation split.

## Notes
* Never commit a GitHub token; if the repository is private, paste a read-only fine-grained token in the configuration cell, or upload `bundle.zip`.
* Different GPUs/library versions change fine-tuned scores slightly; report the GPU, `torch`/`transformers` versions and seeds in the paper.
* `--max-len 512` (A100/L4) changes the experiment (fewer truncated functions); if used, say so and rerun nothing else.
