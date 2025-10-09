import json
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()


# Step 1: Load data
def load_json_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents = []
    for item in data:
        text = f"Q: {item['question']}\nA: {item['answer']}"
        documents.append(Document(page_content=text))
    return documents


json_documents = load_json_file("data/medical_chatbot_data.json")


def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


documents = load_pdf_files("data/")

all_documents = documents + json_documents


# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


text_chunks = create_chunks(extracted_data=all_documents)


# Step 3: Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


embedding_model = get_embedding_model()


# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
