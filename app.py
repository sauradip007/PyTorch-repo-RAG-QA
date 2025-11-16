import streamlit as st
from typing import List
from rag_qa_chatbot import rag_chain, hybrid_retriever  # your backend module

st.set_page_config(page_title="GitHub Issue Resolver", layout="wide")

st.title("🔧 GitHub Issue Resolver (RAG)")
st.caption(
    "Ask anything about Hugging Face Transformers issues. "
    "We preview retrieved GitHub issues and generate a context-aware answer."
)

# -------- Sidebar controls --------
with st.sidebar:
    st.header("⚙️ Options")
    k_preview = st.slider("Preview top-K retrieved issues", min_value=1, max_value=15, value=5, step=1)
    show_snippets = st.checkbox("Show content snippets", value=True)
    st.markdown("---")
    st.info(
        "Tip: Backend uses FAISS (MMR) + BM25 (RRF) with Cohere Rerank for compression. "
        "API keys are loaded in backend via .env."
    )

# -------- Query input --------
query = st.text_input("Enter your query about Hugging Face Transformers:", placeholder="e.g., CUDA OOM with Trainer on bert-base-uncased")

def _render_doc_card(doc, show_content: bool = True):
    meta = getattr(doc, "metadata", {}) or {}
    url = meta.get("url", "N/A")
    title = meta.get("title", "Untitled Issue")
    st.markdown(f"**[{title}]({url})**")
    if show_content:
        snippet = (doc.page_content or "").strip()
        if len(snippet) > 600:
            snippet = snippet[:600] + " ..."
        st.write(snippet)
    with st.expander("View metadata", expanded=False):
        st.json(meta)
    st.markdown("---")

def _preview_retrieval(q: str, k: int):
    try:
        docs = hybrid_retriever.get_relevant_documents(q, k=k)
    except TypeError:
        # Fallback if retriever doesn't accept k kwarg
        docs = hybrid_retriever.get_relevant_documents(q)
        docs = docs[:k] if len(docs) > k else docs
    return docs

# -------- Run pipeline --------
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("🔎 Retrieved GitHub Issues (Preview)")
    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Retrieving relevant issues..."):
                try:
                    docs = _preview_retrieval(query, k_preview)
                    if not docs:
                        st.info("No documents retrieved. Try rephrasing your query.")
                    else:
                        for d in docs:
                            _render_doc_card(d, show_content=show_snippets)
                    st.session_state["last_query"] = query
                except Exception as e:
                    st.error(f"Retrieval error: {e}")

with col_right:
    st.subheader("🧠 RAG Answer")
    run_answer = st.button("Generate Answer", type="primary")
    if run_answer:
        q = query.strip() or st.session_state.get("last_query", "")
        if not q:
            st.warning("Please enter a query (or click Search first).")
        else:
            with st.spinner("Reasoning over retrieved context..."):
                try:
                    answer = rag_chain.invoke(q)  # backend builds {context, query} internally
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"RAG error: {e}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit · FAISS + BM25 (RRF) · Cohere Rerank · OpenAI Chat model")
