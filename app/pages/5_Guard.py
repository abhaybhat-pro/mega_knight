import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.runtime.resources import configure_threads

configure_threads()

import pandas as pd
import streamlit as st

from app.ui import chrome, footer, line_chart
from src.contracts import STREAM_CSV, THRESHOLDS
from src.data.generate import drift_stream, read
from src.erase.queue import load_train_df
from src.guard.monitor import fit_baseline, score_batch, stream_advisories, stream_counts

chrome("Guard")
st.write(
    "Villages 6 and 7 are about 3% of ordinary rows. In a 2-batch window they stay under "
    f"{THRESHOLDS['min_subgroup_size']} rows, so they are marked insufficient data and are never flagged. "
    "Age-band drops are shown and do not add anyone to the hold list. "
    "A held decision waits for a person. The Guard does not refuse care."
)
st.write(
    "Open question: a confidence drop of "
    f"{THRESHOLDS['confidence_drop']} is shown as a warning. "
    "The hold list itself is villages whose PSI reaches "
    f"{THRESHOLDS['psi_major']}."
)
if st.button("Replay the drift stream"):
    try:
        from app.ui import cached_shardset
        ss = cached_shardset()
        train = load_train_df()
        base_rows = read(STREAM_CSV)
        base = fit_baseline(train, ss=ss)
    except Exception as exc:
        st.error(f"Guard replay needs a trained model and the synthetic stream ({type(exc).__name__}: {exc}).")
    else:
        rows = []
        last = None
        reports = []
        for i, batch in enumerate(drift_stream(base_rows)):
            report = score_batch(ss, batch, base)
            reports.append(report)
            last = report
            for item in report.subgroups:
                if not str(item["label"]).startswith("village_"):
                    continue
                rows.append({
                    "batch": i,
                    "village": item["label"],
                    "max_psi": item["max_psi"] if item["max_psi"] is not None else 0.0,
                    "status": item["status"],
                })
        frame = pd.DataFrame(rows)
        line_chart(
            frame, "batch", "max_psi", "village", "Max feature PSI by village",
            hlines=((THRESHOLDS["psi_moderate"], "0.10 moderate"), (THRESHOLDS["psi_major"], "0.25 major")),
        )
        current = frame[frame["batch"] == frame["batch"].max()]
        st.dataframe(current, hide_index=True)
        counts = stream_counts(reports)
        st.session_state["guard_counts"] = counts  # counts only; the Erase page copies them to the receipt
        advisories = stream_advisories(reports)
        st.write(
            f"Stream result (counts only): {counts['drift_flags']} village(s) reached the major drift line; "
            f"{counts['held_decisions']} decision(s) held for human review."
        )
        if advisories:
            st.caption(
                "Advisories, shown but never added to the hold list: "
                + ", ".join(f"{label} ({n} of {len(reports)} batches)" for label, n in advisories.items())
            )
        if last is not None:
            st.session_state["held_ids"] = list(last.held_ids)
            st.session_state["held_villages"] = [
                int(row["label"].split("_")[1])
                for row in last.subgroups
                if row["status"] == "hold-for-review" and row["kind"] == "village"
            ]
            st.subheader("Held for human review (person ids, this session only)")
            st.write(st.session_state["held_ids"] or "Nothing held.")
            st.caption(
                f"Replayed {int(THRESHOLDS['n_batches'])} batches. "
                "Ids are not written to the receipt or the log."
            )
footer()
