#!/usr/bin/env python3
"""Run a trained process-logic transformer on a single prompt.

This is a lightweight inference script for quick checks against a saved
checkpoint produced by train.py. It supports two common actions:

  - next-step ranking: print the top-k candidate next steps
  - continuation generation: sample or greedily decode a continuation

The checkpoint format matches train.py / predict.py:
  {"model": state_dict, "config": preset_name, "vocab_size": int,
   "block_size": int}
"""

from __future__ import annotations

import argparse

import torch

from model import make_model
from vocab import Vocab


def _split_pipe(text: str) -> list[str]:
    return [part for part in (piece.strip() for piece in text.split("|")) if part]


def load_checkpoint(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = make_model(
        ckpt["vocab_size"],
        preset=ckpt["config"],
        block_size=ckpt["block_size"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def prompt_ids(vocab: Vocab, family: str, steps: list[str], device: str):
    ids = vocab.encode(steps, family=family, add_bos=True, add_eos=False)
    return torch.tensor(ids, device=device).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser(description="Run a trained process model")
    ap.add_argument("--ckpt", required=True, help="Path to model.pt")
    ap.add_argument("--vocab", default="vocab.json", help="Path to vocab.json")
    ap.add_argument("--family", required=True, choices=["mosfet", "igbt", "ic"])
    ap.add_argument(
        "--steps",
        required=True,
        help="Prompt steps separated by |, for example 'STEP A|STEP B'",
    )
    ap.add_argument(
        "--mode",
        default="both",
        choices=["both", "next", "generate"],
        help="What to print for the prompt",
    )
    ap.add_argument("--topk", type=int, default=5, help="Top-k next-step candidates")
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=40,
        help="Maximum tokens to generate in continuation mode",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation",
    )
    ap.add_argument(
        "--sample",
        action="store_true",
        help="Sample instead of greedy decoding when generating",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = Vocab.load(args.vocab)
    model, ckpt = load_checkpoint(args.ckpt, device)
    steps = _split_pipe(args.steps)
    idx = prompt_ids(vocab, args.family, steps, device)

    print(f"device={device}")
    print(f"config={ckpt['config']} block_size={ckpt['block_size']} vocab={len(vocab)}")
    print(f"family={args.family}")
    print(f"prompt={'|'.join(steps) if steps else '<empty>'}")

    if args.mode in ("both", "next"):
        top_ids = model.next_step_topk(idx, k=args.topk)
        top_steps = vocab.decode(top_ids, strip_special=True)
        print("next-step candidates:")
        for rank, step in enumerate(top_steps, start=1):
            print(f"  {rank}: {step}")

    if args.mode in ("both", "generate"):
        gen_ids = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            eos_id=vocab.eos,
            temperature=args.temperature,
            greedy=not args.sample,
        )
        gen_steps = vocab.decode(gen_ids, strip_special=True)
        print("continuation:")
        print("|".join(gen_steps))


if __name__ == "__main__":
    main()