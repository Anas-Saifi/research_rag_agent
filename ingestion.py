import hashlib
import os
import re
import unicodedata
from pathlib import Path

import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

load_dotenv()

# bge-large-en-v1.5 has a 512-token limit. Keep chunks well under it, since the
# BGE tokenizer counts differently from tiktoken.
CHUNK_SIZE = 350
CHUNK_OVERLAP = 50
BATCH_SIZE = 128  # chunks per Pinecone upsert


def clean_text(text: str) -> str:
    text = re.sub(r"[\ud800-\udfff]", "", text)  # strip lone surrogates
    return unicodedata.normalize("NFKC", text)


def make_id(doc, i: int) -> str:
    """Deterministic ID so re-running overwrites instead of duplicating."""
    raw = f"{doc.metadata.get('source')}|{doc.metadata.get('page')}|{i}|{doc.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ingestion():
    papers_dir = Path(__file__).resolve().parent / "papers"
    loader = DirectoryLoader(
        path=str(papers_dir), glob="**/*.pdf", loader_cls=PyPDFLoader
    )
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, disallowed_special=()
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding on: {device}")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
    )

    docs = loader.load()
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    print(f"{len(chunks)} chunks created!")

    ids = [make_id(c, i) for i, c in enumerate(chunks)]

    # Output dimension is 1024: the Pinecone index must be 1024-dim and empty
    # (or a fresh index) so Gemini and BGE vectors are never mixed.
    vector_store = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Ingesting"):
        end = start + BATCH_SIZE
        vector_store.add_documents(chunks[start:end], ids=ids[start:end])

    print("Ingestion complete!")


if __name__ == "__main__":
    ingestion()
    print("finished!")