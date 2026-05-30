# Process Logic Transformer

This repository trains and evaluates a small decoder-only transformer for semiconductor process sequences. The model learns word-level process steps for the `mosfet`, `igbt`, and `ic` families, then uses a validator-aware evaluation loop to measure whether generated continuations stay process-valid.

## What’s in the repo

- `train.py` trains the transformer, builds or loads `vocab.json`, and writes checkpoints plus `metrics.jsonl`.
- `predict.py` loads a trained checkpoint and writes submission files for next-step prediction, sequence completion, and anomaly scoring.
- `model.py` contains the GPT-style model and preset sizes (`tiny`, `baseline`, `large`, `xl`).
- `generate_sequences.py` generates and validates synthetic process sequences.
- `validator_tools.py` wraps the validator for full-sequence checks, continuation checks, and negative example generation.
- `baseline_ngram.py` provides a simple trigram baseline for comparison.

## Requirements

- Python 3.10+
- PyTorch

The scripts use only the standard library plus `torch`.

## Data format

Training data is expected as CSV files containing sequences in the long format used by `generate_sequences.py` and `train.py`. The scripts infer the product family from the filename, so files should include names such as `mosfet`, `igbt`, or `ic`.

## Train

Example:

```bash
python train.py --config baseline \
  --data data/mosfet.csv data/igbt.csv data/ic.csv \
  --heldout data/heldout.csv \
  --out ckpt
```

This will:

- build `vocab.json` if it does not already exist,
- train the transformer,
- log metrics to `ckpt/metrics.jsonl`,
- save the checkpoint to `ckpt/model.pt`.

Useful flags:

- `--config`: model preset from `model.py`
- `--epochs`: number of epochs to train
- `--batch_size`: batch size
- `--block_size`: maximum context length
- `--dropout`: dropout rate

## Predict

After training, run inference with the saved checkpoint:

```bash
python predict.py --ckpt ckpt/model.pt \
  --vocab vocab.json \
  --valid eval_input_valid.csv \
  --anomaly eval_input_anomaly.csv \
  --out submission
```

This writes:

- `nextstep.csv`
- `completion.csv`
- `anomaly.csv`

## Baseline

If you want a non-neural baseline for comparison:

```bash
python baseline_ngram.py --train data/mosfet.csv data/igbt.csv data/ic.csv \
  --valid eval_input_valid.csv \
  --out submission_baseline
```

## Notes

- `train.py` logs both standard loss/accuracy and the process-logic continuation metric.
- `predict.py` expects the official evaluation CSVs from the event, so verify column names against the provided scorer if needed.
- The repository already includes a sample `model.pt` and `metrics.jsonl`.