import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.runtime.resources import configure_threads

configure_threads()

import streamlit as st

from app.ui import chrome, footer
from src.contracts import CANARY_COL, ID_COL
from src.erase.queue import erase, load_train_df
from src.model.shards import shard_of
from src.report.receipt import (
    receipt_to_html, verify_log, verify_receipt, verify_receipt_log_linkage,
)

chrome("Erase")
st.write("One shard is retrained. The receipt is HMAC-signed and appended to the hash chain.")


def _verify_box(receipt: dict) -> None:
    from src.report.receipt import get_hmac_key
    try:
        key = get_hmac_key()
    except Exception as exc:
        st.error(f"Set FORGETME_HMAC_KEY before erasing ({exc}).")
        return
    from src.contracts import LOG_PATH
    sig_ok = verify_receipt(receipt, key)
    chain_ok, _bad = verify_log(LOG_PATH)
    linked, why = verify_receipt_log_linkage(receipt, LOG_PATH)
    if sig_ok and chain_ok and linked:
        st.metric("Receipt", "✔ VERIFIED")
    else:
        st.metric("Receipt", "not verified")
        st.info(f"INVALID (expected a matching signature and log link: {why})")
    st.json({k: receipt[k] for k in receipt if k != "hmac"})
    st.download_button("Download printable receipt", receipt_to_html(receipt),
                       file_name="receipt.html", mime="text/html")


if st.button("Erase one recorded person"):
    try:
        from app.ui import cached_shardset
        ss = cached_shardset()
        frame = load_train_df()
        ordinary = frame[frame[CANARY_COL] != "member"] if CANARY_COL in frame.columns else frame
        if ordinary.empty:
            st.error("The training set has no ordinary row to erase.")
        else:
            pid = int(ordinary.iloc[0][ID_COL])
            guard_counts = st.session_state.get("guard_counts")  # set by the Guard page; counts only
            with st.spinner("Retraining the affected shard and signing the receipt"):
                receipt = erase([pid], "REQ-UI-1", ss, frame, guard_summary=guard_counts)
            st.session_state["last_receipt"] = receipt
            st.write(f"Erased one ordinary record from shard {shard_of(pid, ss.n_shards)}.")
            if guard_counts is None:
                st.caption("No Guard replay in this session, so the receipt records the Guard stream counts as null (not recorded).")
            else:
                st.caption("The receipt carries the Guard replay counts from this session (counts only, no person ids).")
            _verify_box(receipt)
    except Exception as exc:
        st.error(f"Erase did not finish ({type(exc).__name__}: {exc}).")

if st.button("Tamper-test the receipt"):
    receipt = st.session_state.get("last_receipt")
    if not receipt:
        st.info("INVALID (expected a receipt to tamper). Erase a record first.")
    else:
        flipped = dict(receipt)
        digest = flipped.get("hmac", "")
        flipped["hmac"] = ("0" if not digest else ("1" if digest[0] != "1" else "2")) + digest[1:]
        st.session_state["tampered"] = flipped
        st.info("INVALID (expected the signature check to fail after one changed byte).")
        from src.report.receipt import get_hmac_key
        try:
            ok = verify_receipt(flipped, get_hmac_key())
        except Exception:
            ok = False
        st.write("Tampered signature accepted:" , ok)

footer()
