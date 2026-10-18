import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.runtime.resources import configure_threads

configure_threads()

import pandas as pd
import streamlit as st

from app.ui import bar_chart, chrome, feature_form, footer
from src.contracts import FEATURES, THRESHOLDS
from src.model.shards import predict

chrome("Predict")
values = feature_form()
held_villages = st.session_state.get("held_villages", [])
if int(values["village_code"]) in held_villages:
    st.warning(
        "HELD FOR REVIEW: the Guard flagged this village. A clinician should confirm this decision.",
        icon="⚠️",
    )
if st.button("Score this visit"):
    try:
        from app.ui import cached_shardset
        ss = cached_shardset()
        frame = pd.DataFrame([{name: values[name] for name in FEATURES}])
        out = predict(ss, frame)
    except FileNotFoundError as exc:
        st.error(f"No trained ensemble yet ({exc}). Train shards first.")
    else:
        proba = float(out["proba"][0])
        st.metric("Referral probability", f"{proba:.3f}")
        st.metric("Label", int(out["label"][0]))
        st.metric("Confidence", f"{float(out['confidence'][0]):.3f}")
        st.metric("Entropy (bits)", f"{float(out['entropy'][0]):.3f}")
        labels = [str(k) for k in sorted(out["per_shard"])]
        heights = [float(out["per_shard"][int(k)][0]) for k in labels]
        bar_chart(labels, heights, "Per-shard referral probability", "Shard",
                  "Referral probability", hline=float(THRESHOLDS["referral_threshold"]))
footer()
