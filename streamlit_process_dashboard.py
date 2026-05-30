#!/usr/bin/env python3
"""
streamlit_process_dashboard.py

Bad-ass Streamlit dashboard for the semiconductor process prediction model.

Expected files in the same project folder:
- model.py
- vocab.py
- ckpt/model.pt
- vocab.json

Run:
    pip install streamlit pandas torch plotly
    streamlit run streamlit_process_dashboard.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from vocab import Vocab
from model import make_model


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Process Logic AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
        border: 1px solid rgba(148,163,184,0.25);
        box-shadow: 0 10px 30px rgba(0,0,0,0.22);
    }
    .good {
        color: #22c55e;
        font-weight: 800;
    }
    .warn {
        color: #f59e0b;
        font-weight: 800;
    }
    .bad {
        color: #ef4444;
        font-weight: 800;
    }
    .step-box-given {
        padding: 0.55rem 0.75rem;
        border-radius: 12px;
        background: rgba(59,130,246,0.15);
        border: 1px solid rgba(59,130,246,0.35);
        margin-bottom: 0.35rem;
        font-size: 0.9rem;
    }
    .step-box-pred {
        padding: 0.55rem 0.75rem;
        border-radius: 12px;
        background: rgba(249,115,22,0.15);
        border: 1px solid rgba(249,115,22,0.35);
        margin-bottom: 0.35rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

SPECIAL_TOKENS = {"<PAD>", "<BOS>", "<EOS>", "<UNK>", "<MOSFET>", "<IGBT>", "<IC>"}


def split_pipe_sequence(text: str) -> List[str]:
    return [x.strip() for x in text.strip().split("|") if x.strip()]


def clean_steps(steps: List[str]) -> List[str]:
    return [s.strip().upper() for s in steps if s and s.strip()]


@st.cache_resource(show_spinner=False)
def load_vocab_and_model(ckpt_path: str, vocab_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = Vocab.load(vocab_path)

    model = make_model(
        ckpt["vocab_size"],
        preset=ckpt["config"],
        block_size=ckpt["block_size"],
        dropout=0.0,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, vocab, ckpt, device


def prompt_ids(vocab: Vocab, steps: List[str], family: str, device: str):
    ids = vocab.encode(steps, family=family.lower(), add_bos=True, add_eos=False)
    return torch.tensor(ids, device=device).unsqueeze(0)


@torch.no_grad()
def next_step_topk(model, vocab: Vocab, steps: List[str], family: str, device: str, k: int = 5) -> List[Tuple[str, float]]:
    idx = prompt_ids(vocab, steps, family, device)

    # Prefer model helper if available
    if hasattr(model, "next_step_topk"):
        top_ids = model.next_step_topk(idx, k=max(k + 5, 10))
        decoded = vocab.decode(top_ids, strip_special=True)
        result = []
        for token in decoded:
            if token not in SPECIAL_TOKENS and token not in [x[0] for x in result]:
                result.append((token, 0.0))
            if len(result) == k:
                break
        return result

    # Generic fallback
    out = model(idx)
    logits = out[0] if isinstance(out, tuple) else out
    last_logits = logits[:, -1, :]
    probs = torch.softmax(last_logits, dim=-1)[0]
    values, indices = torch.topk(probs, k=max(k + 10, 20))

    result = []
    for prob, token_id in zip(values, indices):
        token = vocab.decode([int(token_id.item())], strip_special=True)
        if isinstance(token, list):
            token = token[0] if token else ""
        if token and token not in SPECIAL_TOKENS and token not in [x[0] for x in result]:
            result.append((token, float(prob.item())))
        if len(result) == k:
            break
    return result


@torch.no_grad()
def complete_sequence(model, vocab: Vocab, steps: List[str], family: str, device: str, max_new: int = 80) -> List[str]:
    idx = prompt_ids(vocab, steps, family, device)
    gen_ids = model.generate(idx, max_new_tokens=max_new, eos_id=vocab.eos)

    # model.generate may return either:
    # - torch.Tensor shaped [1, tokens]
    # - list[int]
    # - list[list[int]]
    if isinstance(gen_ids, torch.Tensor):
        ids = gen_ids[0].tolist() if gen_ids.ndim == 2 else gen_ids.tolist()
    elif isinstance(gen_ids, list):
        if gen_ids and isinstance(gen_ids[0], list):
            ids = gen_ids[0]
        else:
            ids = gen_ids
    else:
        raise TypeError(f"Unsupported generate() output type: {type(gen_ids)}")

    full_steps = vocab.decode(ids, strip_special=True)

    # keep only continuation if full sequence was returned
    if len(full_steps) >= len(steps) and full_steps[: len(steps)] == steps:
        continuation = full_steps[len(steps) :]
    else:
        # fallback: remove prefix length anyway
        continuation = full_steps[len(steps) :] if len(full_steps) > len(steps) else full_steps

    return [s for s in continuation if s not in SPECIAL_TOKENS]


@torch.no_grad()
def sequence_nll(model, vocab: Vocab, steps: List[str], family: str, device: str) -> float:
    ids = vocab.encode(steps, family=family.lower(), add_bos=True, add_eos=True)
    idx = torch.tensor(ids, device=device).unsqueeze(0)
    x, y = idx[:, :-1], idx[:, 1:]

    out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    logp = torch.log_softmax(logits, dim=-1)
    tok_lp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return float(-tok_lp.mean().item())


def health_from_nll(nll: float) -> Tuple[int, str]:
    # Rough heuristic; tune this later with labeled validation data.
    if nll < 0.8:
        return 96, "Low risk"
    if nll < 1.5:
        return 78, "Medium risk"
    if nll < 2.5:
        return 55, "High risk"
    return 25, "Very high risk"


def make_timeline(prefix: List[str], continuation: List[str]) -> pd.DataFrame:
    rows = []
    for i, step in enumerate(prefix, start=1):
        rows.append({"index": i, "step": step, "type": "Given"})
    for j, step in enumerate(continuation, start=len(prefix) + 1):
        rows.append({"index": j, "step": step, "type": "Predicted"})
    return pd.DataFrame(rows)


def read_eval_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

st.sidebar.title("⚙️ Model Control")

ckpt_path = st.sidebar.text_input("Checkpoint path", "ckpt/model.pt")
vocab_path = st.sidebar.text_input("Vocab path", "vocab.json")
family = st.sidebar.selectbox("Product family", ["MOSFET", "IGBT", "IC"])
max_new = st.sidebar.slider("Max completion steps", 10, 160, 80, 5)
top_k = st.sidebar.slider("Top-K next steps", 3, 10, 5, 1)

load_button = st.sidebar.button("Load model", type="primary")

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if load_button:
    st.cache_resource.clear()
    st.session_state.model_loaded = False

try:
    model, vocab, ckpt, device = load_vocab_and_model(ckpt_path, vocab_path)
    st.session_state.model_loaded = True
except Exception as e:
    st.session_state.model_loaded = False
    model = vocab = ckpt = device = None
    st.sidebar.error(f"Model not loaded: {e}")

if st.session_state.model_loaded:
    st.sidebar.success(f"Loaded on {device}")
    st.sidebar.caption(f"Config: {ckpt.get('config')} | Block size: {ckpt.get('block_size')} | Vocab: {ckpt.get('vocab_size')}")


# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------

st.markdown('<div class="main-title">⚙️ Semiconductor Process Logic AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict next fabrication steps, complete partial process flows, and score process anomalies.</div>',
    unsafe_allow_html=True,
)

if not st.session_state.model_loaded:
    st.warning("Load a valid checkpoint and vocab file from the sidebar first.")
    st.stop()


tab_predict, tab_csv, tab_anomaly, tab_about = st.tabs([
    "🔮 Predict",
    "📄 Eval CSV Runner",
    "🚨 Anomaly Score",
    "ℹ️ Model Info",
])


# -----------------------------------------------------------------------------
# Predict tab
# -----------------------------------------------------------------------------

with tab_predict:
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.subheader("Input partial sequence")
        input_mode = st.radio("Input format", ["Pipe separated", "One step per line"], horizontal=True)

        default_text = (
            "RECEIVE WAFER LOT|LOT IDENTIFICATION|INITIAL WAFER INSPECTION|"
            "MEASURE SURFACE DEFECTS|WAFER CLEAN PRE PROCESS|RCA CLEAN 1|WET CLEAN RCA2|HF DIP|DRY WAFER"
        )

        raw = st.text_area("Paste process prefix", default_text, height=220)

        if input_mode == "Pipe separated":
            prefix_steps = split_pipe_sequence(raw)
        else:
            prefix_steps = [line.strip() for line in raw.splitlines() if line.strip()]

        prefix_steps = clean_steps(prefix_steps)

        run = st.button("🚀 Predict process", type="primary")

    with col_right:
        st.subheader("Current prefix")
        st.metric("Given steps", len(prefix_steps))
        with st.expander("Show prefix steps", expanded=False):
            for step in prefix_steps:
                st.markdown(f'<div class="step-box-given">{step}</div>', unsafe_allow_html=True)

    if run:
        if not prefix_steps:
            st.error("Paste at least one step.")
        else:
            top = next_step_topk(model, vocab, prefix_steps, family, device, k=top_k)
            continuation = complete_sequence(model, vocab, prefix_steps, family, device, max_new=max_new)
            nll = sequence_nll(model, vocab, prefix_steps + continuation, family, device)
            health, risk = health_from_nll(nll)

            st.divider()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Next step", top[0][0] if top else "N/A")
            m2.metric("Predicted remaining", len(continuation))
            m3.metric("Process health", f"{health}%")
            m4.metric("NLL anomaly score", f"{nll:.3f}")

            c1, c2 = st.columns([0.9, 1.1], gap="large")

            with c1:
                st.subheader("Top next-step predictions")
                if top:
                    df_top = pd.DataFrame(top, columns=["step", "probability"])
                    if df_top["probability"].sum() == 0:
                        df_top["confidence"] = [100 / len(df_top)] * len(df_top)
                    else:
                        df_top["confidence"] = df_top["probability"] * 100
                    fig = px.bar(df_top, x="confidence", y="step", orientation="h", text="confidence")
                    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig.update_layout(height=360, yaxis={"categoryorder": "total ascending"}, xaxis_title="Confidence %", yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader("Predicted continuation")
                if continuation:
                    cont_df = pd.DataFrame({"#": range(1, len(continuation) + 1), "Predicted step": continuation})
                    st.dataframe(cont_df, use_container_width=True, height=360)
                else:
                    st.info("No continuation generated.")

            st.subheader("Process timeline")
            tl = make_timeline(prefix_steps, continuation)
            fig_tl = px.scatter(
                tl,
                x="index",
                y="type",
                color="type",
                hover_data=["step"],
                title="Given prefix vs predicted continuation",
            )
            fig_tl.update_traces(marker=dict(size=12))
            fig_tl.update_layout(height=320, xaxis_title="Step index", yaxis_title="")
            st.plotly_chart(fig_tl, use_container_width=True)

            with st.expander("Export predicted continuation"):
                predicted_pipe = "|".join(continuation)
                st.code(f"EXAMPLE_ID,PREDICTED_SEQUENCE\ntest_0001,{predicted_pipe}", language="csv")


# -----------------------------------------------------------------------------
# CSV runner tab
# -----------------------------------------------------------------------------

with tab_csv:
    st.subheader("Run official eval-style CSV")
    st.caption("Expected columns: EXAMPLE_ID,FAMILY,COMPLETION_FRACTION,PARTIAL_SEQUENCE")

    uploaded = st.file_uploader("Upload eval_input_valid.csv", type=["csv"])

    if uploaded:
        df = read_eval_csv(uploaded)
        st.dataframe(df.head(20), use_container_width=True)

        if st.button("Generate nextstep.csv + completion.csv", type="primary"):
            next_rows = []
            comp_rows = []
            progress = st.progress(0)

            for i, row in df.iterrows():
                ex_id = row["EXAMPLE_ID"]
                fam = str(row["FAMILY"]).strip().upper()
                steps = clean_steps(split_pipe_sequence(str(row["PARTIAL_SEQUENCE"])))

                top = next_step_topk(model, vocab, steps, fam, device, k=5)
                top_steps = [x[0] for x in top]
                while len(top_steps) < 5:
                    top_steps.append("")

                continuation = complete_sequence(model, vocab, steps, fam, device, max_new=max_new)

                next_rows.append({
                    "EXAMPLE_ID": ex_id,
                    "RANK_1": top_steps[0],
                    "RANK_2": top_steps[1],
                    "RANK_3": top_steps[2],
                    "RANK_4": top_steps[3],
                    "RANK_5": top_steps[4],
                })
                comp_rows.append({
                    "EXAMPLE_ID": ex_id,
                    "PREDICTED_SEQUENCE": "|".join(continuation),
                })
                progress.progress((i + 1) / len(df))

            next_df = pd.DataFrame(next_rows)
            comp_df = pd.DataFrame(comp_rows)

            st.success("Generated predictions")
            st.subheader("nextstep.csv")
            st.dataframe(next_df, use_container_width=True)
            st.download_button("Download nextstep.csv", next_df.to_csv(index=False).encode("utf-8"), "nextstep.csv", "text/csv")

            st.subheader("completion.csv")
            st.dataframe(comp_df, use_container_width=True)
            st.download_button("Download completion.csv", comp_df.to_csv(index=False).encode("utf-8"), "completion.csv", "text/csv")


# -----------------------------------------------------------------------------
# Anomaly tab
# -----------------------------------------------------------------------------

with tab_anomaly:
    st.subheader("Sequence anomaly score")
    st.caption("This uses model negative log-likelihood. Higher = more surprising / more suspicious.")

    anomaly_text = st.text_area(
        "Paste full or partial sequence",
        "RECEIVE WAFER LOT|LOT IDENTIFICATION|INITIAL WAFER INSPECTION|MEASURE SURFACE DEFECTS",
        height=200,
        key="anomaly_text",
    )

    threshold = st.slider("Suspicion threshold", 0.1, 5.0, 1.5, 0.1)

    if st.button("Score anomaly"):
        steps = clean_steps(split_pipe_sequence(anomaly_text))
        if not steps:
            st.error("Paste a sequence first.")
        else:
            nll = sequence_nll(model, vocab, steps, family, device)
            health, risk = health_from_nll(nll)
            is_valid = nll < threshold

            c1, c2, c3 = st.columns(3)
            c1.metric("NLL score", f"{nll:.3f}")
            c2.metric("Health", f"{health}%")
            c3.metric("Status", "VALID ✅" if is_valid else "SUSPICIOUS 🚨")

            if is_valid:
                st.success(f"Sequence looks normal. Risk: {risk}")
            else:
                st.error(f"Sequence looks suspicious. Risk: {risk}")

            st.info("For final hackathon scoring, tune the threshold on labeled valid/invalid examples.")


# -----------------------------------------------------------------------------
# About tab
# -----------------------------------------------------------------------------

with tab_about:
    st.subheader("Model information")

    info = {
        "Device": device,
        "Checkpoint": ckpt_path,
        "Vocabulary": vocab_path,
        "Config": ckpt.get("config"),
        "Block size": ckpt.get("block_size"),
        "Vocab size from checkpoint": ckpt.get("vocab_size"),
        "Loaded vocab length": len(vocab),
    }

    st.json(info)

    st.markdown(
        """
        ### What this dashboard demonstrates

        - **Task 1:** top-5 next-step prediction  
        - **Task 2:** sequence completion until `SHIP LOT` or max token limit  
        - **Task 3:** anomaly scoring using model likelihood  

        ### Important

        The anomaly score is useful, but for the final challenge you should tune the threshold with known valid and generated invalid examples.
        """
    )
