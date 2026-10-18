import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.runtime.resources import configure_threads

configure_threads()

import streamlit as st

from app.ui import chrome, footer
from src.contracts import N_SHARDS
from src.data.generate import generate
from src.erase.queue import load_train_df
from src.model.shards import train_all
from src.runtime.resources import available_cores

chrome("Train")
st.write(
    "Differential privacy and shard unlearning are different guarantees. "
    "This page trains the shard ensemble. It does not train a differentially private model."
)
st.write(f"Cores visible to this process: {available_cores()}. Shard count: {N_SHARDS}.")
if st.button("Generate synthetic cohort"):
    with st.spinner("Writing the synthetic CSVs"):
        generate()
    st.success("Synthetic cohort written.")
if st.button("Train shards"):
    try:
        frame = load_train_df()
    except OSError as exc:
        st.error(f"No training file yet ({exc}). Generate the cohort first.")
    else:
        with st.spinner("Training the ensemble"):
            ss = train_all(frame)
        st.session_state["ss_ready"] = True
        rows = {k: v.get("rows") for k, v in ss.manifest.get("shards", {}).items()}
        st.write("Rows per shard", rows)
        st.write("Ensemble fingerprint", ss.manifest.get("ensemble_hash"))
footer()
