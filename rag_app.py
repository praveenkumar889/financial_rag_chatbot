import streamlit as st
import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.getcwd())

# Load environment variables explicitly
load_dotenv(override=True)

from rag_core.engine import answer_question

st.set_page_config(
    page_title="Financial Assistant",
    layout="wide"
)

st.title("📊 Financial Assistant")
st.caption("Adaptive • Self-correcting • Source-grounded")

# Session memory
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question (FY24–FY25):")

if st.button("Ask") and query:
    with st.spinner("Retrieving, validating, and self-correcting..."):
        try:
            result = answer_question(query)
            st.session_state.history.append((query, result))
        except Exception as e:
            st.error(f"Error: {e}")

# Display chat history (latest first)
for q, res in reversed(st.session_state.history):
    st.markdown(f"### ❓ {q}")
    st.markdown(res["answer"])

    if res["confidence"] == "High":
        st.success(f"Confidence: High ({res['score']:.2f})")
    elif res["confidence"] == "Medium":
        st.warning(f"Confidence: Medium ({res['score']:.2f})")
    else:
        st.error("Confidence: Low")

    if res["confidence"] != "Low" and res["sources"]:
        with st.expander("📚 Sources"):
            for s in res["sources"]:
                st.write(s)

    st.divider()
