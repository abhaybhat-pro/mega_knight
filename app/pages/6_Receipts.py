import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.runtime.resources import configure_threads

configure_threads()

import streamlit as st

from app.ui import chrome, footer
from src.contracts import LOG_PATH
from src.report.receipt import verify_log

chrome("Receipts")
path = Path(LOG_PATH)
if not path.is_file():
    st.info("No erasure log yet.")
else:
    ok, bad = verify_log(str(path))
    st.metric("Log chain", "intact" if ok else f"broken at {bad}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        rows.append({k: entry.get(k) for k in ("seq", "request_id", "timestamp_utc", "shard_ids")})
    st.dataframe(rows, hide_index=True)
    st.caption("The log stores counts and hashes. It does not store clinical features.")
footer()
