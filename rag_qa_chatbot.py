from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
from langchain_community.document_loaders import GitHubIssuesLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.output_parsers import PydanticOutputParser
import os
import pickle
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.llms import Cohere
# import torch





load_dotenv()


# class IssueAnswer(BaseModel):
#     summary: str = Field(..., description="A concise summary of the issue in 2-3 sentences.")
#     root_cause: Optional[str] = Field(None, description="Likely root cause of the issue if available.")
#     fixes: List[str] = Field(default_factory=list, description="List of fixes or workarounds explicitly suggested in the context.")
#     code_snippets: List[str] = Field(default_factory=list, description="Relevant code blocks, commands, or error logs extracted from the context.")
#     references: List[str] = Field(default_factory=list, description="List of GitHub issue URLs cited from the context.")
#     confidence: Optional[str] = Field(None, description="Low/Medium/High confidence in the provided fix, based on context coverage.")
#     notes: Optional[str] = Field(None, description="Any additional notes, limitations, or caveats mentioned in the context.")


def format_docs(docs):
    formatted = []
    for doc in docs:
     url = doc.metadata.get("url", "N/A")
     title = doc.metadata.get("title", "Untitled Issue")
     formatted.append(f"[{title}]({url})\n\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)



# Configuration
BASE_DIR = os.path.dirname(__file__)
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "faiss_vector_store")
BM25_RETRIEVER_PATH = os.path.join(BASE_DIR, "bm25_retriever.pkl")
REBUILD_VECTOR_STORE = False  # Set to True if you want to rebuild the vector store

# Check if vector store already exists
if os.path.exists(VECTOR_STORE_PATH) and not REBUILD_VECTOR_STORE:
    print("Loading existing vector store...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully!")
    
    # Try to load existing BM25 retriever
    if os.path.exists(BM25_RETRIEVER_PATH):
        print("Loading existing BM25 retriever...")
        with open(BM25_RETRIEVER_PATH, 'rb') as f:
            bm25_retriever = pickle.load(f)
        print("BM25 retriever loaded successfully!")
    else:
        print("BM25 retriever not found. Creating from FAISS docstore...")
        # Rebuild BM25 from all documents in FAISS docstore
        try:
            all_docs = list(getattr(vector_store.docstore, "_dict", {}).values())
            if not all_docs:
                # fallback: sample some docs via similarity search
                all_docs = vector_store.similarity_search("transformers", k=200)
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            with open(BM25_RETRIEVER_PATH, 'wb') as f:
                pickle.dump(bm25_retriever, f)
            print("BM25 retriever created from docstore and saved successfully!")
        except Exception as e:
            print(f"Failed to rebuild BM25 from docstore: {e}")
            # last resort: create empty BM25 to avoid crashes
            bm25_retriever = BM25Retriever.from_documents([])
else:
    print("Building new vector store...")
    
    # Step 1: Load the data
    print("Loading GitHub issues...")
    loader = GitHubIssuesLoader(repo="huggingface/transformers", include_prs=False)
    data = loader.load()
    data = data[:200]  # Limit to 200 documents for faster processing
    print(f"Loaded {len(data)} documents")

    # Step 2: Initialize embeddings and semantic chunker
    print("Initializing embeddings and semantic chunker...")
    embeddings = OpenAIEmbeddings()
    text_splitter = SemanticChunker(
        embeddings=embeddings, 
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=3
    )

    # Step 3: Apply semantic chunking to ALL documents
    print("Applying semantic chunking to all documents...")
    all_chunks = []
    for i, doc in enumerate(data):
        print(f"Processing document {i+1}/{len(data)}")
        chunks = text_splitter.split_documents([doc])
        all_chunks.extend(chunks)
    print(f"Created {len(all_chunks)} chunks from {len(data)} documents")

    # Step 4: Create vector store
    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(all_chunks, embeddings,metadata_keys=["url","title"])
    
    # Step 5: Save vector store to disk
    print("Saving vector store to disk...")
    vector_store.save_local(VECTOR_STORE_PATH)
    
    # Step 6: Create BM25 retriever and save it
    print("Creating BM25 retriever...")
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    
    # Save BM25 retriever
    with open(BM25_RETRIEVER_PATH, 'wb') as f:
        pickle.dump(bm25_retriever, f)
    print("BM25 retriever created and saved successfully!")
    
    print("Vector store saved successfully!")

print("Setting up retrieval chain...")

# creating rer

# Create retrievers
faiss_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k":5,"lambda_mult":0.7})


compressor = CohereRerank(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=faiss_retriever
)


# Create hybrid retriever
try:
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[0.5, 0.5],
        search_type="rrf"
    )
    print("Hybrid retriever created successfully!")
    
except Exception as e:
    print(f"Error creating hybrid retriever: {e}")
    print("Falling back to FAISS retriever only...")
    hybrid_retriever = faiss_retriever

print("Creating RAG chain...")

# parser = PydanticOutputParser(pydantic_object=IssueAnswer)
parser = StrOutputParser()
# Prompt template
prompt = PromptTemplate(
    template = """
You are a GitHub issue resolver. 
A user has asked the following question about the Hugging Face Transformers library:

User query:
{query}

You are also given context from one or more GitHub issues related to this query:

Context:
{context}

Instructions:
- Use the context to suggest the most likely cause and a possible fix or workaround.
- Summarize the root cause in simple terms.
- If fixes or workarounds are provided in the context, explain them clearly.
- If multiple possible fixes are given, list them as options.
- If no clear fix is available in the context, say that the issue may still be unresolved and provide the GitHub issue link for updates.
- Try to always include the GitHub issue number or URL from the metadata for reference.
- Do not invent solutions beyond what is supported by the context.


""",
input_variables = ['context','query']
# partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
# parser = StrOutputParser()


parallel_chain = RunnableParallel({
    "context": compression_retriever | RunnableLambda(format_docs), 
    "query": RunnablePassthrough()
})

rag_chain = parallel_chain | prompt | llm | parser


if __name__ == "__main__":
# Test query
    test_query = "When I try to fine-tune bert-base-uncased using Trainer, I get CUDA out of memory even with a small batch size. How can I fix this?"

# Debug: Let's see what documents are being retrieved
    print("=== DEBUGGING RETRIEVAL ===")
    print(f"Original query: {test_query}")

# Get retrieved documents
    retrieved_docs = hybrid_retriever.get_relevant_documents(test_query)
    print(f"\nRetrieved {len(retrieved_docs)} documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Document {i+1} ---")
        print(f"Content preview: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")

    print("\n=== RAG RESPONSE ===")
    print(rag_chain.invoke(test_query))
    
#  __all__ = ["rag_chain", "IssueAnswer", "hybrid_retriever"]




# print(embeddings.embed_documents([chunks[0].page_content]))

# vectore_store = FAISS.from_documents(chunks, embeddings)
