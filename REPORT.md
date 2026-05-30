# Team 123123 — Industrial AI (Infineon)

## Team

- **Volodymyr Sennikov** — ML / pipeline / evaluation
- **Ronald Juhasz** — ML / training / infrastructure
- **Stefan Ilic** — ML / tooling / dashboard

**Track:** Industrial AI (Infineon) — Learning and Benchmarking Process Logic

---

## TL;DR

We train a small decoder-only transformer **from scratch** on self-generated
semiconductor process sequences, with each process step as a single token. A
rule validator is wired directly into the evaluation loop so we measure real
process-logic validity, not just loss. Against a trigram baseline we show the
transformer reaches ~0.83 held-out token accuracy and produces process-valid
continuations, while the baseline collapses on unseen families.

---

## Problem

The task is to learn the *logic* of semiconductor process routes — long ordered
sequences of ~107–151 steps across three families (MOSFET, IGBT, IC) — rather
than memorizing surface patterns. Concretely we target the three submission
tasks: next-step prediction, sequence completion, and anomaly detection, plus
the hidden 4th-family OOD generalization the organizers score separately.

The central question we kept honest about: does the model learn the rules that
generate valid sequences, or does it just memorize n-grams? We built our whole
evaluation around answering that with numbers.

---

## Approach

- **Decoder-only transformer trained from scratch**, word-level vocabulary (1
  process step = 1 token). No pretrained LLM: the vocabulary is ~200 tokens, so
  a pretrained model's natural-language machinery and subword tokenizer would
  only hurt.
- **Number normalization in the tokenizer.** Steps like `ALIGN MASK LEVEL 1..6`
  were originally separate tokens, so the model treated each level as unrelated.
  We split a trailing integer into a shared base token plus a `<Nk>` number
  token. This let the model generalize "same step, different level" while still
  seeing the number for ordering rules. Measured effect: held-out token accuracy
  0.808 → 0.828, val loss 0.332 → 0.300, with no change to model size.
- **Validator wired into the eval loop.** Every epoch we let the model complete
  held-out prefixes and use the rule validator to measure the fraction of
  continuations that introduce no new violation — a "process-logic" signal
  separate from loss. We also generate labeled rule-violating sequences for
  anomaly detection.
- **Trigram baseline as the floor and the OOD evidence engine.** A pure
  frequency-counting model tells us what "good" means, and its collapse on an
  unseen family is our evidence that local statistics don't transfer.
- **Anomaly detection via model likelihood with a threshold tuned on labeled
  data** (never on the eval set), so we avoid silently assuming a fixed anomaly
  rate.

Where it runs: training and inference on the Leonardo cluster (1 node, A100s,
SLURM); data generation and the baseline run on CPU.

---

## How to run it

See `README.md` for full detail. Core flow:

```bash
pip install -r requirements.txt

# 1. generate data (or use training_data/*.csv)
python generate_sequences.py --family mosfet --count 5000 --output data/mosfet.csv --seed 1
#    ... igbt seed 2, ic seed 3; held-out with different seeds ...

# 2. build the vocabulary once
python vocab.py data/*.csv

# 3. baseline (CPU, the floor to beat)
python baseline_ngram.py --train data/*.csv --heldout data/heldout_ic.csv

# 4. train the transformer (GPU, via SLURM on Leonardo)
sbatch train.slurm

# 5. produce submission files (Task 3 threshold tuned on labeled data)
python predict.py --ckpt ckpt/model.pt --vocab vocab.json \
    --valid eval_input_valid.csv --anomaly eval_input_anomaly.csv \
    --task3_labeled training_data/MOSFET_variants.csv training_data/IGBT_variants.csv training_data/IC_variants.csv \
    --out submission

# 6. interactive demo
streamlit run streamlit_process_dashboard.py
```

---

## Results

Baseline vs. transformer, held-out (in-distribution):

| Metric | Trigram baseline | Transformer (baseline cfg, 3.3M) |
| --- | --- | --- |
| next-step / token accuracy | 0.77 | **0.83** |
| process-valid continuations | n/a | **~1.0** (validator-checked) |
| vocab-normalization effect | — | +2pp token acc, −10% val loss |

OOD (train on two families, evaluate on the unseen third):

| Metric | Trigram baseline | Transformer |
| --- | --- | --- |
| next-step / token accuracy (unseen family) | 0.40 | ~0.47 |

Submission files (`submission/nextstep.csv`, `completion.csv`, `anomaly.csv`)
are produced by `predict.py` against the official eval inputs and scored with the
organizers' `eval_metrics.py`. Training artifacts: `ckpt/model.pt`,
`metrics.json` (per-epoch loss / token-accuracy / process-logic curve).

---

## What worked

- **Number normalization** was the single cleanest win: a tokenizer change, not
  a bigger model, moved held-out token accuracy and loss measurably.
- **Validator-in-the-loop** gave us a process-logic metric independent of loss —
  the model reaches ~100% valid continuations on held-out prefixes.
- **The baseline** kept us honest: it showed next-step top-5 is saturated (~0.99
  for everyone, not a useful metric) and that completion exact-match is ~0 by
  the nature of the metric — so we didn't over-read a scary-looking 0.

## What didn't work

- **Scaling the model didn't help token accuracy.** Going from 3.3M to 25.5M
  parameters on 90k sequences left held-out token accuracy at the same ~0.83
  plateau. The ceiling is the task's inherent unpredictability (optional steps),
  not model capacity.
- **OOD generalization is weak.** On an unseen family the transformer (~0.47) is
  only marginally above the trigram baseline (0.40), and its held-out loss rose
  over training. We do not claim OOD as a win — it's an honest negative result.

## What we'd do with another 36 hours

- Randomize the optional-step probabilities and cycle counts during data
  generation, so the model isn't calibrated to a single marginal — likely the
  biggest lever for OOD robustness.
- Train a **discriminative** anomaly head on labeled valid/invalid examples
  instead of relying on generative likelihood, which is sensitive to marginals.
- Run the full scaling sweep (tiny/baseline/large/xl) as four parallel
  single-GPU jobs and report a proper scaling curve.

---

## Track-specific deliverables (Industrial AI / Infineon)

- [x] Eval submission files in `submission/`: `nextstep.csv`, `completion.csv`, `anomaly.csv`
- [x] Training artifacts: `ckpt/model.pt`, `metrics.json` (loss + token-acc + process-logic curves)
- [ ] Scores from `eval_metrics.py` on all three tasks (run with the organizers'
      ground-truth files; per-family breakdown printed by the script)
- [x] Demo shows baseline vs. trained output on identical inputs (Streamlit dashboard)

---

## Credits & dependencies

- **Libraries:** PyTorch, NumPy, pandas, Plotly, Streamlit (see `requirements.txt`)
- **Pre-trained models:** none — trained from scratch
- **External APIs:** none
- **Datasets:** organizers' synthetic process data + sequences generated by the
  provided `generate_sequences.py`
- **Infrastructure:** Leonardo (EuroHPC / CINECA), SLURM, pixi environment
- **AI coding assistance:** Claude (Anthropic) used during development

---

*Submitted by team 123123 for the Industrial AI track, 2026.*
