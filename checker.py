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

load_dotenv()

embeddings = OpenAIEmbeddings()

VECTOR_STORE_PATH = "./faiss_vector_store"
vector_store = FAISS.load_local(
    VECTOR_STORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)


# Retrieve one sample doc (doesn't need a query, just use similarity with a dummy vector)
docs = vector_store.similarity_search("test", k=1)

print("=== SAMPLE DOCUMENT FROM VECTOR STORE ===")

print("=== DOCUMENT ===")
print(docs[0])
print("=== Page Content ===")
print("Page Content:\n", docs[0].page_content[:500], "...\n") 
print("=== Metadata ===")
 # preview first 500 chars
print("Metadata:\n", docs[0].metadata)

__all__ = ["rag_chain", "IssueAnswer"]

