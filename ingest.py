# ingest.py
import os
import fitz  # PyMuPDF
from openai import OpenAI
from pinecone import Pinecone
import uuid
from dotenv import load_dotenv

# Init OpenAI + Pinecone
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.environ.get("PINECONE_INDEX", "math-tutor-index")

index = pc.Index(INDEX_NAME)  # make sure index has dimension=1536

# ---- Helpers ----
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_texts(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in response.data]

# ---- Ingest ----
def ingest_documents():
    documents_folder = "documents"
    for filename in os.listdir(documents_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(documents_folder, filename)
            text = extract_text_from_pdf(file_path)
            chunks = chunk_text(text)

            embeddings = embed_texts(chunks)

            # Upsert into Pinecone
            vectors = [
                (str(uuid.uuid4()), embeddings[i], {"text": chunks[i]})
                for i in range(len(chunks))
            ]
            index.upsert(vectors=vectors)

    print("Ingestion completed!")

if __name__ == "__main__":
    ingest_documents()
