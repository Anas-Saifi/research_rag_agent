from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import re
import unicodedata
def ingestion():
    loader = DirectoryLoader(path = "C:/Anas/Others/langchain_practice_projects/resumeproject1/papers", glob = "**/*.pdf", loader_cls = PyPDFLoader)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=10000, chunk_overlap=250, disallowed_special= ())
    embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001", output_dimensionality = 1024)
    docs = loader.load()
    for doc in docs:
        text = doc.page_content
        text = re.sub(r'[\ud800-\udfff]', '', text)
        text = unicodedata.normalize("NFKC", text)
        doc.page_content = text
    chunks = splitter.split_documents(docs)

    print(f"{len(chunks)} chunks created!")
    vector_store = PineconeVectorStore.from_documents(index_name = os.environ["INDEX_NAME"], embedding = embeddings, documents = chunks)

    print("Ingestion complete!")

if __name__ == "__main__":
    ingestion()
    print("finished!")