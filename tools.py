import urllib, urllib.request
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from dotenv import load_dotenv
load_dotenv()

import hashlib
from pathlib import Path
import feedparser
BASE_DIR = Path(__file__).resolve().parent
SEARCH_PAPERS_DIR = BASE_DIR / "search_papers"
SEARCH_PAPERS_DIR.mkdir(parents=True, exist_ok=True)

embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001", output_dimensionality = 1024)
vector_store = PineconeVectorStore(index_name = os.environ["INDEX_NAME"], embedding = embeddings)



@tool
def search_tool(query: str):
    """
    Searches for the the relevant papers in the arxiv database and ingests them to the vector database
    args: query of the user
    returns: document id of the ingested documents

    """
    query = query.replace(" ", "+")
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=1'
    data = urllib.request.urlopen(url)
    raw_xml = data.read().decode('utf-8')
    vfeed = feedparser.parse(raw_xml)
    if not vfeed.entries:
        return "No paper found for this query"
    entry = vfeed.entries[0]
    paper_title = entry.title.strip()
    pdf_link = entry.id.replace("abs", "pdf")
    
    # Sanitize title to prevent invalid filename characters on Windows/Linux
    import re
    safe_title = re.sub(r'[^\w\-_\. ]', '_', paper_title)[:120]
    file_path = SEARCH_PAPERS_DIR / f"{safe_title}.pdf"

    urllib.request.urlretrieve(pdf_link, str(file_path))
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 1000, chunk_overlap = 100, disallowed_special = ())   
    chunks = splitter.split_documents(docs)

    document_id = hashlib.sha256(
        file_path.read_bytes()
    ).hexdigest()
    for i, doc in enumerate(chunks):
        doc.metadata["document_id"] = document_id
        doc.metadata["file_name"] = file_path.name
        doc.metadata["source"] = str(file_path)
        doc.metadata["chunk_id"] = i
    vector_store.add_documents(documents = chunks)
    
    return document_id


@tool
def retrieve_chunks(query: str, document_id = None):
    """
    Retrieve relevant documents from the vector database.
    args:
        query: query of the user
        document_id: document id to be passed only if the retrieved document was freshly ingested by the search_tool, leave otherwise
        returns:
            list of relevant papers
    """
    filter_dict = {"document_id": document_id} if document_id else None
    result = vector_store.similarity_search_with_score(query, k = 5, filter=filter_dict)
    if not result:
        return "no relevant content found"
    docs = []
    for doc, score in result:
        docs.append(doc.page_content)

    return docs






if __name__ == "__main__":
    search_tool("")
    print("finished!")