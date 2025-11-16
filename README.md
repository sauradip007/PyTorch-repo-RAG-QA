# GitHub Issue Resolver (RAG)

RAG app that searches HuggingFace/Transformers GitHub issues using a hybrid retriever (BM25 + FAISS + Cohere rerank) and answers queries with an LLM.

## Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # put your keys
