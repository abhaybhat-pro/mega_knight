import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.runtime.resources import configure_threads

configure_threads()

from app.ui import chrome, footer
from pathlib import Path as _Path

chrome("ForgetMe + Aegis-Z Guard")
st_path = _Path(ROOT) / "docs" / "ibm_status.md"
import streamlit as st

st.write(
    "A clinic honours a forget-me request by retraining one shard of an 8-shard triage model. "
    "A membership-inference audit supplies before-and-after evidence. The audit is not a proof "
    "that the person is forgotten. Aegis-Z Guard holds a risky decision for a human. "
    "It does not refuse care."
)
st.subheader("IBM status")
if st_path.is_file():
    st.markdown(st_path.read_text(encoding="utf-8"))
else:
    st.info("docs/ibm_status.md is not in this tree yet.")
st.caption("Granite, Triton, the Z Deep Learning Compiler, ONNX and differential privacy stay Planned until a log says they ran. Telum, Spyre and Secure Execution are not claimed.")
footer()
