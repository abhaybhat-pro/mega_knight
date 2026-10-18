import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.runtime.resources import configure_threads

configure_threads()

import streamlit as st

from app.ui import chrome, footer

chrome("Audit")
st.write(
    "Shard attack AUC and ensemble attack AUC are both shown, with 95% intervals. "
    "The ensemble number is lower because most shards never saw the erased person. "
    "That dilution is by design: the guarantee being checked is the shard that held the records."
)
st.write("The audit is empirical evidence from this attack, not a proof of forgetting.")
receipt = st.session_state.get("last_receipt")
if not receipt:
    st.info("Erase a member canary to see a before/after audit on the receipt. This page does not invent numbers.")
else:
    for label, block in (("Before", receipt.get("audit_before")), ("After", receipt.get("audit_after"))):
        st.subheader(label)
        if not isinstance(block, dict):
            st.write("No audit block on this receipt.")
            continue
        for level in ("shard", "ensemble"):
            view = block.get(level)
            if not isinstance(view, dict):
                st.write(f"{level}: not recorded")
                continue
            ci = view.get("auc_ci95")
            st.write(
                f"{level} AUC {view.get('attack_auc')} "
                f"interval {ci} raw {view.get('attack_auc_raw')} "
                f"TPR at 1% FPR {view.get('tpr_at_1pct_fpr')}"
            )
footer()
