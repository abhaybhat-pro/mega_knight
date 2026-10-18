import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.runtime.resources import configure_threads

configure_threads()

import streamlit as st

from app.ui import chrome, footer, read_csv

chrome("Benchmarks")
st.caption("Figures below are whatever is on disk. A laptop or WSL run is not an s390x result.")
results = read_csv("bench/results.csv")
if results is None:
    st.info("bench/results.csv is not available yet.")
else:
    st.subheader("FR-10 timings")
    st.dataframe(results, hide_index=True)
curve = read_csv("bench/shards_curve/results.csv")
if curve is None:
    st.info("bench/shards_curve/results.csv is not available yet.")
else:
    st.subheader("Shard-count curve")
    st.dataframe(curve, hide_index=True)
profile = read_csv("bench/resource_profile.csv")
if profile is None:
    st.info("bench/resource_profile.csv is not available yet.")
else:
    st.subheader("Resource profile")
    st.dataframe(profile, hide_index=True)
footer()
