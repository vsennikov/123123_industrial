# Process Logic Transformer — Team 123123 (Industrial AI / Infineon)

A small decoder-only transformer that learns semiconductor process sequences for
the `mosfet`, `igbt`, and `ic` families. Each process step is one token. A rule
validator is wired into evaluation so we measure whether generated continuations
stay process-valid — not just loss.

See `REPORT.md` for the technical write-up, results, and honest findings.

## Repository layout

```
.
├── README.md                  ← this file
├── REPORT.md                  ← technical report (jury reads this)
├── LICENSE                    ← MIT
├── requirements.txt
│
├── vocab.py                   ← tokenizer (word-level + number normalization)
├── model.py                   ← GPT-style model, presets tiny/baseline/large/xl
├── train.py                   ← training loop + validator-in-the-loop eval
├── predict.py                 ← writes submission files (nextstep/completion/anomaly)
├── baseline_ngram.py          ← trigram baseline (the floor to beat)
├── validator_tools.py         ← validator wrapper + labeled-negative generator
├── generate_sequences.py      ← organizers' sequence generator + validator
├── run_model.py               ← quick single-prompt inference
├── streamlit_process_dashboard.py  ← interactive demo (baseline vs trained)
│
├── train.slurm                ← Leonardo job (1 GPU)
├── train_ood.slurm            ← Leonardo job (OOD experiment)
│
├── vocab.json                 ← prebuilt vocabulary (200 tokens)
├── ckpt/model.pt              ← trained checkpoint
├── metrics.json               ← per-epoch training metrics
│
├── training_data/             ← pre-generated sequences (MOSFET/IGBT/IC)
├── eval_input_valid.csv       ← official eval input (Tasks 1 & 2)
├── eval_input_anomaly.csv     ← official eval input (Task 3)
└── submission/                ← nextstep.csv, completion.csv, anomaly.csv
```

## Requirements

- Python 3.10+
- PyTorch (training / inference)
- pandas, plotly, streamlit (dashboard only)

```bash
pip install -r requirements.txt
```

The core scripts (`train.py`, `predict.py`, `vocab.py`, `model.py`,
`baseline_ngram.py`, `validator_tools.py`) use only the standard library + torch.
pandas/plotly/streamlit are needed only for the dashboard.

## Data format

Long format (`SEQUENCE_ID, STEP`, one row per step), as produced by
`generate_sequences.py`. The scripts infer the family from the filename, so names
must contain `mosfet`, `igbt`, or `ic`.

## Train

```bash
python train.py --config baseline \
  --data data/mosfet.csv data/igbt.csv data/ic.csv \
  --heldout data/heldout_mosfet.csv data/heldout_igbt.csv data/heldout_ic.csv \
  --vocab vocab.json --out ckpt --epochs 30
```

Builds `vocab.json` if missing, trains, logs to `ckpt/metrics.jsonl`, saves
`ckpt/model.pt`. On Leonardo: `sbatch train.slurm`.

Held-out files must use **different seeds** than training — they are our
memorization detector.

## Baseline (run this first)

```bash
python baseline_ngram.py --train data/*.csv --heldout data/heldout_ic.csv
```

Prints the floor (next-step top-1/top-5, completion exact-match). Run the OOD
variant (train on two families, test on the third) to see it collapse — that gap
is our generalization evidence.

## Predict (submission files)

```bash
python predict.py --ckpt ckpt/model.pt --vocab vocab.json \
  --valid eval_input_valid.csv --anomaly eval_input_anomaly.csv \
  --task3_labeled training_data/MOSFET_variants.csv training_data/IGBT_variants.csv training_data/IC_variants.csv \
  --out submission
```

`--task3_labeled` is **required** for anomaly detection — it tunes the threshold
on our own labeled data, never on the eval set.

Score with the organizers' script:

```bash
python eval_metrics.py --task next-step  --ground-truth <gt> --predictions submission/nextstep.csv
python eval_metrics.py --task completion --ground-truth <gt> --predictions submission/completion.csv
python eval_metrics.py --task anomaly    --ground-truth <gt> --predictions submission/anomaly.csv
```

## Interactive demo

```bash
streamlit run streamlit_process_dashboard.py
```

Loads `ckpt/model.pt` + `vocab.json`. Shows next-step prediction, sequence
completion, and anomaly scoring on a prompt — useful for the baseline-vs-trained
comparison in the demo video.

## Notes on running on Leonardo

- Install the environment on a **login node** (compute nodes have no internet).
- Use `$SCRATCH` for data and checkpoints during the hackathon.
- `train.slurm` requests 1 GPU under the hackathon reservation.