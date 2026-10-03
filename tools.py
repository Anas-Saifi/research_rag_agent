import hashlib
import os
import re
import urllib.request
from functools import lru_cache
from pathlib import Path

import feedparser
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SEARCH_PAPERS_DIR = BASE_DIR / "search_papers"
SEARCH_PAPERS_DIR.mkdir(parents=True, exist_ok=True)

# Must match the settings used when the index was built (ingestion.py).
CHUNK_SIZE = 350
CHUNK_OVERLAP = 50


@lru_cache(maxsize=1)
def get_vector_store() -> PineconeVectorStore:
    """Load the embedding model on first use, not at import time.

    Importing torch/sentence-transformers and loading bge-large is slow, and
    doing it at import blocks the web server from opening its port, which is
    what makes Render report "No open ports detected".
    """
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-large-en-v1.5",
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
    )
    return PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )


@tool
def search_tool(query: str):
    """
    Searches for the the relevant papers in the arxiv database and ingests them to the vector database
    args: query of the user
    returns: document id of the ingested documents

    """
    query = query.replace(" ", "+")
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=1"
    with urllib.request.urlopen(url, timeout=30) as data:
        raw_xml = data.read().decode("utf-8")
    vfeed = feedparser.parse(raw_xml)
    if not vfeed.entries:
        return "No paper found for this query"
    entry = vfeed.entries[0]
    paper_title = entry.title.strip()
    pdf_link = entry.id.replace("abs", "pdf")

    # Sanitize title to prevent invalid filename characters on Windows/Linux
    safe_title = re.sub(r"[^\w\-_\. ]", "_", paper_title)[:120]
    file_path = SEARCH_PAPERS_DIR / f"{safe_title}.pdf"

    urllib.request.urlretrieve(pdf_link, str(file_path))
    docs = PyPDFLoader(str(file_path)).load()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, disallowed_special=()
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]

    document_id = hashlib.sha256(file_path.read_bytes()).hexdigest()
    for i, doc in enumerate(chunks):
        doc.metadata["document_id"] = document_id
        doc.metadata["file_name"] = file_path.name
        doc.metadata["source"] = str(file_path)
        doc.metadata["chunk_id"] = i
    get_vector_store().add_documents(documents=chunks)

    return document_id


@tool
def retrieve_chunks(query: str, document_id: str | None = None):
    """
    Retrieve relevant documents from the vector database.
    args:
        query: query of the user
        document_id: document id to be passed only if the retrieved document was freshly ingested by the search_tool, leave otherwise
        returns:
            list of relevant papers
    """
    filter_dict = {"document_id": document_id} if document_id else None
    result = get_vector_store().similarity_search_with_score(
        query, k=5, filter=filter_dict
    )
    if not result:
        return "no relevant content found"
    return [doc.page_content for doc, score in result]