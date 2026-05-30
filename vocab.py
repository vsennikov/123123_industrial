#!/usr/bin/env python3
"""
vocab.py — word-level tokenizer with NUMBER NORMALIZATION.

Why number normalization:
  Steps like "ALIGN MASK LEVEL 1..6", "EXPOSE LITHO LEVEL 1..6", "RCA CLEAN 1/2"
  used to each become a SEPARATE token. That bloated the vocab (~205) and, worse,
  made the model treat "ALIGN MASK LEVEL 3" and "ALIGN MASK LEVEL 4" as unrelated
  symbols -- so it couldn't generalize "this is the same step at a different
  level". token_acc stalled around 0.81 as a result.

  Fix: split a trailing integer into its own token.
      "ALIGN MASK LEVEL 3"   -> base "ALIGN MASK LEVEL"  + number token "<N3>"
      "EXPOSE LITHO LEVEL 2"  -> base "EXPOSE LITHO LEVEL" + "<N2>"
  All levels now share ONE base token, while the number survives as a separate
  token so the model can still learn ordering rules like RULE_LITHO_LEVEL_SKIP.

  encode() expands each step into [base, <Nk>] (or just [step] if no number).
  decode() re-joins a base followed by a number token back into "BASE k", so the
  output is IDENTICAL to the original step string. This roundtrip is essential
  for the validator and for the submission format.
"""

from __future__ import annotations
import json
import re
from pathlib import Path

SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
FAMILY_TOKENS = {"mosfet": "<MOSFET>", "igbt": "<IGBT>", "ic": "<IC>"}
NUMBER_TOKENS = [f"<N{i}>" for i in range(1, 10)]

_TRAILING_NUM = re.compile(r"^(.*?) (\d+)$")


def split_step(step: str):
    m = _TRAILING_NUM.match(step)
    if m:
        return m.group(1), m.group(2)
    return step, None


def join_step(base: str, num):
    return f"{base} {num}" if num is not None else base


class Vocab:
    def __init__(self, stoi):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}

    @property
    def pad(self): return self.stoi["<PAD>"]
    @property
    def bos(self): return self.stoi["<BOS>"]
    @property
    def eos(self): return self.stoi["<EOS>"]
    @property
    def unk(self): return self.stoi["<UNK>"]

    def __len__(self): return len(self.stoi)

    def encode(self, steps, family=None, add_bos=True, add_eos=True):
        ids = []
        if add_bos:
            ids.append(self.bos)
        if family is not None:
            ids.append(self.stoi[FAMILY_TOKENS[family.lower()]])
        for s in steps:
            base, num = split_step(s)
            ids.append(self.stoi.get(base, self.unk))
            if num is not None:
                ids.append(self.stoi.get(f"<N{num}>", self.unk))
        if add_eos:
            ids.append(self.eos)
        return ids

    def decode(self, ids, strip_special=True):
        structural = set(SPECIAL) | set(FAMILY_TOKENS.values())
        toks = [self.itos.get(int(i), "<UNK>") for i in ids]
        out = []
        i = 0
        while i < len(toks):
            t = toks[i]
            if strip_special and t in structural:
                i += 1
                continue
            if t in NUMBER_TOKENS:
                i += 1
                continue
            if i + 1 < len(toks) and toks[i + 1] in NUMBER_TOKENS:
                num = toks[i + 1][2:-1]
                out.append(join_step(t, num))
                i += 2
            else:
                out.append(t)
                i += 1
        return out

    def save(self, path):
        Path(path).write_text(json.dumps(self.stoi, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def build_vocab(step_iter):
    bases = set()
    for s in step_iter:
        base, _ = split_step(s)
        bases.add(base)
    tokens = (list(SPECIAL) + list(FAMILY_TOKENS.values())
              + NUMBER_TOKENS + sorted(bases))
    stoi = {t: i for i, t in enumerate(tokens)}
    return Vocab(stoi)


if __name__ == "__main__":
    import sys, csv
    steps = []
    for p in sys.argv[1:]:
        with open(p, encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            col = "STEP" if "STEP" in r.fieldnames else r.fieldnames[-1]
            for row in r:
                steps.append(row[col])
    v = build_vocab(steps)
    v.save("vocab.json")
    print(f"vocab size = {len(v)}  (saved to vocab.json)")