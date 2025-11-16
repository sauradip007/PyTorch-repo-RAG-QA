# 🔧 GitHub Issue Resolver (RAG)

RAG app that searches HuggingFace/Transformers GitHub issues using a **hybrid retriever** (BM25 + FAISS + Cohere rerank) and answers queries with an LLM.

---

## 🚀 Quick Start (TL;DR)

```bash
git clone https://github.com/<your-username>/github-issue-resolver-rag.git
cd github-issue-resolver-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your API keys
streamlit run app.py
