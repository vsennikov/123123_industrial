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
process-logic validity, not just loss. On 450 held-out completion examples the
transformer more than halves the trigram baseline's normalized edit distance
(0.22 vs 0.55) and beats it on token accuracy (0.42 vs 0.32); for anomaly
detection it reaches F1 = 0.96 on labeled data.

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
python3 generate_sequences.py --family mosfet --count 5000 --output data/mosfet.csv --seed 1
#    ... igbt seed 2, ic seed 3; held-out with different seeds ...

# 2. build the vocabulary once
python3 vocab.py data/*.csv

# 3. baseline (CPU, the floor to beat)
python3 baseline_ngram.py --train data/*.csv --heldout data/heldout_ic.csv

# 4. train the transformer (GPU, via SLURM on Leonardo)
sbatch train.slurm

# 5. produce submission files (Task 3 threshold tuned on labeled data)
python3 predict.py --ckpt ckpt/model.pt --vocab vocab.json \
    --valid eval_input_valid.csv --anomaly eval_input_anomaly.csv \
    --task3_labeled training_data/MOSFET_variants.csv training_data/IGBT_variants.csv training_data/IC_variants.csv \
    --out submission

# 6. interactive demo
streamlit run streamlit_process_dashboard.py
```

---

## Results

**Sequence completion (Task 2), 450 held-out examples, in-distribution**
(generated with held-out seeds the model never trained on), scored with the same
metrics as the official `eval_metrics.py`:

| Metric | Transformer (3.3M) | Trigram baseline |
| --- | --- | --- |
| Normalized Edit Distance (lower better) | **0.2218** | 0.5466 |
| Token accuracy | **0.4201** | 0.3229 |
| Exact match | 0.0000 | 0.0033 |

Per-family NED (transformer vs baseline): MOSFET 0.165 / 0.505, IGBT 0.226 / 0.562,
IC 0.274 / 0.573 — the transformer wins on every family. NED is the main
completion metric, and the transformer more than halves it. Exact match is ~0 for
both, as expected for completing 40–60 steps where any single deviation breaks it.

**Anomaly detection (Task 3):** model-likelihood scoring with a threshold tuned on
labeled data (never on the eval set) reaches **F1 = 0.958** on our labeled
validation set. Applied to the official `eval_input_anomaly.csv` (987 sequences),
it produces `submission/anomaly.csv`.

**Next-step / token accuracy (in-distribution held-out):** transformer ~0.83 vs
trigram 0.77. Top-5 is saturated (~0.99 for both) and is not a discriminating metric.

**Number-normalization ablation:** collapsing `LEVEL N` steps into a shared base
token plus a `<Nk>` number token improved held-out token accuracy 0.808 → 0.828
and val loss 0.332 → 0.300, at identical model size.

Submission files (`submission/nextstep.csv`, `completion.csv`, `anomaly.csv`) are
produced by `predict.py` against the official eval inputs. Training artifacts:
`ckpt/model.pt`, `metrics.json`.

---

## What worked

- **Completion clearly beats the baseline.** On 450 held-out examples the
  transformer's NED (0.22) is less than half the trigram's (0.55), and it wins on
  every family. This is our strongest evidence the model learned real continuation
  logic rather than local frequencies.
- **Anomaly detection is strong.** F1 = 0.96 on labeled data, with the threshold
  tuned on labeled examples (never on the eval set).
- **Number normalization** was a clean, measurable win: a tokenizer change, not a
  bigger model, moved held-out token accuracy 0.808 → 0.828 and loss 0.332 → 0.300.
- **Validator-in-the-loop** gave us a process-logic metric independent of loss.
- **The baseline** kept us honest: it showed next-step top-5 is saturated (~0.99
  for everyone) and exact-match is ~0 by the metric's nature — so we didn't
  over-read a scary-looking 0.

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