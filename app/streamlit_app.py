import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from rag.pipeline import RAGPipeline

st.set_page_config(page_title="Tringa's RAG Chatbot", page_icon="🛒", layout="wide")

st.markdown(
    """
    <style>
    /* Background */
    .stApp { background-color: #545d38; }

    /* Titles */
    h1, h2, h3 { color: #2f5d50; }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 10px;
    }

    /* Buttons */
    button {
        background-color: #6b8f71 !important;
        color: white !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Tech in Food Supply Chain - RAG Chatbot ")
st.caption(
    "Welcome to Tringa's Chatbot :) ! Ask questions about technology in the food supply chain. "
    "The chatbot answers strictly from uploaded documents and clearly indicates when an answer cannot be found."
)

with st.sidebar:
    st.header("Index")
    st.write("Put docs in `data/raw/` and run ingestion:")
    st.code("python -m rag.ingest --data_dir data/raw --index_dir data/index")
    st.write("Then restart this app.")
    show_debug = st.toggle("Show retrieved chunks (debug)", value=False)

    st.divider()
    doc_filter = st.text_input("Filter by document name (optional)", "")

@st.cache_resource
def get_pipeline():
    return RAGPipeline(index_dir="data/index")

# Load pipeline once
try:
    rag = get_pipeline()
except Exception as e:
    st.error("Index not found or failed to load. Run ingestion first.")
    st.exception(e)
    st.stop()

tab_chat, tab_dashboard = st.tabs(["Chat", "Dashboard"])

# Chat
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask a question about your documents…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        result = rag.ask(question, doc_filter=doc_filter.strip() or None)

        with st.chat_message("assistant"):
            st.markdown(result["answer"])

            if result.get("sources"):
                st.markdown("**Sources:**")
                for s in result["sources"]:
                    st.markdown(
                        f"- {s['doc_name']} — p.{s['page']} c.{s['chunk_id']} (score {s['score']:.3f})"
                    )

            if show_debug and result.get("retrieved"):
                with st.expander("Retrieved chunks"):
                    for r in result["retrieved"]:
                        st.markdown(
                            f"**{r['doc_name']} p.{r['page']} c.{r['chunk_id']} — score {r['score']:.3f}**"
                        )
                        st.write(r["text"])

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

# Dashboard
with tab_dashboard:
    import json
    import pandas as pd

    st.subheader("Observability")
    log_file = "logs/rag_logs.jsonl"

    if not os.path.exists(log_file):
        st.info("No logs yet. Ask a few questions first.")
    else:
        rows = [json.loads(l) for l in open(log_file, "r", encoding="utf-8") if l.strip()]
        df = pd.DataFrame(rows)

        st.dataframe(df, use_container_width=True)
        st.metric("Total queries", len(df))
        st.metric("Unknown rate", f"{df['unknown'].mean()*100:.1f}%")
        st.bar_chart(df["latency_s"])
