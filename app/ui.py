"""Shared Streamlit chrome. Plotly is imported here only, with a matplotlib fallback."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.resources import configure_threads

configure_threads()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.contracts import FEATURES, REPO_ROOT
from src.model.shards import load

FOOTER = "synthetic data; not legal advice; demonstration, not a medical device"
_MARKERS = ["o", "s", "D", "^", "v", "<", ">", "P"]


def chrome(title: str) -> None:
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    st.caption(FOOTER)


def footer() -> None:
    st.divider()
    st.caption(FOOTER)


@st.cache_resource(show_spinner=False)
def cached_shardset():
    """The served ensemble only. Shadow models are never loaded into the UI."""
    return load()


def read_csv(path: str):
    file = Path(path)
    if not file.is_file():
        file = Path(REPO_ROOT) / path
    if not file.is_file():
        return None
    return pd.read_csv(file)


def line_chart(frame: pd.DataFrame, x: str, y: str, color: str, title: str, hlines=()) -> None:
    st.subheader(title)
    try:
        import plotly.express as px

        fig = px.line(frame, x=x, y=y, color=color, markers=True, title=title)
        for value, label in hlines:
            fig.add_hline(y=value, line_dash="dash", annotation_text=label)
        st.plotly_chart(fig, use_container_width=True)
        return
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (name, part) in enumerate(frame.groupby(color)):
        ax.plot(part[x], part[y], marker=_MARKERS[i % len(_MARKERS)], label=str(name))
    for value, label in hlines:
        ax.axhline(value, ls="--", label=label)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    st.pyplot(fig)
    plt.close(fig)


def bar_chart(labels, values, title: str, xlabel: str, ylabel: str, hline=None) -> None:
    st.subheader(title)
    try:
        import plotly.express as px

        fig = px.bar(x=labels, y=values, title=title, labels={"x": xlabel, "y": ylabel})
        if hline is not None:
            fig.add_hline(y=hline, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
        return
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(labels, values)
    if hline is not None:
        ax.axhline(hline, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)


def feature_form() -> dict:
    cols = st.columns(3)
    values = {}
    defaults = {
        "age": 52, "sex": 0, "bmi": 27.0, "sbp": 120.0, "dbp": 80.0, "hba1c": 5.6,
        "smoker": 0, "sym_fatigue": 0, "sym_vision": 0, "sym_thirst": 0,
        "sym_chest_pain": 0, "village_code": 3, "prior_visits": 1,
    }
    for i, name in enumerate(FEATURES):
        current = defaults[name]
        with cols[i % 3]:
            if isinstance(current, int):
                values[name] = st.number_input(name, value=int(current), step=1)
            else:
                values[name] = st.number_input(name, value=float(current))
    return values
