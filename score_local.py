#!/usr/bin/env python3
"""
score_local.py — honest local scoring on YOUR OWN held-out data.

Why this exists:
  The official eval_metrics.py needs ground-truth files (FULL_SEQUENCE) that the
  organizers did NOT give us. So we can't score the official eval locally. But we
  CAN score on our own held-out sequences (where we know the full sequence), using
  the SAME metrics eval_metrics.py uses (normalized edit distance, exact match,
  token accuracy, block accuracy). This gives the real numbers for REPORT.md and a
  fair transformer-vs-baseline comparison.

Usage:
    python score_local.py --ckpt ckpt/model.pt --vocab vocab.json \
        --heldout data/heldout_mosfet.csv data/heldout_igbt.csv data/heldout_ic.csv \
        --train training_data/MOSFET_variants.csv training_data/IGBT_variants.csv training_data/IC_variants.csv

  --heldout : full sequences the model never trained on (cut here at 60/80%)
  --train   : sequences to fit the trigram baseline on (for the comparison)

Runs on CPU fine (small model). Prints a transformer-vs-baseline table.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import torch

from vocab import Vocab
from model import make_model
from generate_sequences import read_csv_sequences
from baseline_ngram import NGramModel, load as load_ngram


# --- metrics (mirrors eval_metrics.py) ------------------------------------- #

def levenshtein(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_edit_distance(pred, ref):
    if not pred and not ref:
        return 0.0
    return levenshtein(pred, ref) / max(len(pred), len(ref), 1)


def token_accuracy(pred, ref):
    if not ref:
        return 1.0 if not pred else 0.0
    n = min(len(pred), len(ref))
    correct = sum(1 for i in range(n) if pred[i] == ref[i])
    return correct / len(ref)


# --- family inference from filename ---------------------------------------- #

def family_of(path):
    n = Path(path).name.lower()
    for fam in ("mosfet", "igbt", "ic"):
        if fam in n:
            return fam
    return "unk"


def load_heldout(paths):
    out = []
    for p in paths:
        fam = family_of(p)
        for steps in read_csv_sequences(Path(p)).values():
            out.append((fam, steps))
    return out


# --- transformer wrappers -------------------------------------------------- #

def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    m = make_model(ck["vocab_size"], preset=ck["config"], block_size=ck["block_size"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def tf_complete(model, vocab, steps, family, device, max_new=120):
    ids = vocab.encode(steps, family=family, add_bos=True, add_eos=False)
    idx = torch.tensor(ids, device=device).unsqueeze(0)
    gen = model.generate(idx, max_new_tokens=max_new, eos_id=vocab.eos)
    return vocab.decode(gen, strip_special=True)


# --- scoring --------------------------------------------------------------- #

def score(name, complete_fn, samples, cut_fracs=(0.6, 0.8)):
    ned, exact, tacc = [], [], []
    ned_by = {}
    for fam, steps in samples:
        for frac in cut_fracs:
            cut = max(1, int(len(steps) * frac))
            prefix, ref = steps[:cut], steps[cut:]
            pred = complete_fn(fam, prefix)
            ned.append(normalized_edit_distance(pred, ref))
            exact.append(pred == ref)
            tacc.append(token_accuracy(pred, ref))
            ned_by.setdefault(fam, []).append(normalized_edit_distance(pred, ref))
    mean = lambda L: sum(L) / len(L) if L else float("nan")
    print(f"\n=== {name} ===")
    print(f"  Mean Normalized Edit Distance : {mean(ned):.4f}  (lower better)")
    print(f"  Exact Match Rate              : {mean(exact):.4f}")
    print(f"  Mean Token Accuracy           : {mean(tacc):.4f}")
    print("  NED by family:")
    for fam in sorted(ned_by):
        print(f"    {fam:<7s} {mean(ned_by[fam]):.4f}")
    return {"ned": mean(ned), "exact": mean(exact), "tacc": mean(tacc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", default="vocab.json")
    ap.add_argument("--heldout", nargs="+", required=True)
    ap.add_argument("--train", nargs="+", required=True,
                    help="sequences to fit the trigram baseline on")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    vocab = Vocab.load(args.vocab)
    model = load_model(args.ckpt, device)
    held = load_heldout(args.heldout)
    print(f"held-out sequences: {len(held)}")

    # transformer
    tf_fn = lambda fam, prefix: tf_complete(model, vocab, prefix, fam, device)
    tf = score("TRANSFORMER (held-out completion)", tf_fn, held)

    # baseline
    ng = NGramModel().fit(load_ngram(args.train))
    bl_fn = lambda fam, prefix: ng.complete(fam, prefix)
    bl = score("TRIGRAM BASELINE (held-out completion)", bl_fn, held)

    print("\n=== SUMMARY (for REPORT) ===")
    print(f"{'metric':<22s} {'transformer':>12s} {'baseline':>12s}")
    print(f"{'NED (lower better)':<22s} {tf['ned']:>12.4f} {bl['ned']:>12.4f}")
    print(f"{'exact match':<22s} {tf['exact']:>12.4f} {bl['exact']:>12.4f}")
    print(f"{'token accuracy':<22s} {tf['tacc']:>12.4f} {bl['tacc']:>12.4f}")


if __name__ == "__main__":
    main()
