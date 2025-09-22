# ingest.py
import os
import re
import fitz  # PyMuPDF
from openai import OpenAI
from pinecone import Pinecone
import uuid
import json
from dotenv import load_dotenv

# Init OpenAI + Pinecone
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = 'math-tutor-index3'  # os.environ.get("PINECONE_INDEX", "math-tutor-index")

index = pc.Index(INDEX_NAME)  # ensure dimension=1536 for text-embedding-3-small


# ---- Helpers ----
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF (preserves layout better than PyPDF2)."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


def chunk_by_problems(text: str, max_len: int = 2000):
    """
    Split text by 'Problem X' markers.
    Falls back to character-based splitting if a problem is too long.
    """
    raw_chunks = re.split(r"(?=Problem\s+\d+)", text, flags=re.IGNORECASE)
    raw_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]

    processed_chunks = []
    for chunk in raw_chunks:
        if len(chunk) > max_len:
            # Secondary split if a single problem is very long
            for i in range(0, len(chunk), max_len):
                processed_chunks.append(chunk[i:i+max_len])
        else:
            processed_chunks.append(chunk)
    return processed_chunks


def embed_texts(texts):
    """Create embeddings for a list of texts."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in response.data]


# ---- Ingest ----
def ingest_documents():
    documents_folder = "documents"

    # Load existing IDs if file exists
    ids_file = "ids.json"
    if os.path.exists(ids_file):
        with open(ids_file, "r") as f:
            all_ids = json.load(f)
    else:
        all_ids = []

    for filename in os.listdir(documents_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(documents_folder, filename)
            print(f"Processing {file_path}...")

            text = extract_text_from_pdf(file_path)
            chunks = chunk_by_problems(text)

            print(f" → Extracted {len(chunks)} chunks")

            embeddings = embed_texts(chunks)

            vectors = []
            for i in range(len(chunks)):
                vector_id = str(uuid.uuid4())
                vectors.append((vector_id, embeddings[i], {"text": chunks[i]}))
                all_ids.append(vector_id)  # Save ID to list

            # Upsert into Pinecone
            index.upsert(vectors=vectors)

    # Save updated IDs to file
    with open(ids_file, "w") as f:
        json.dump(all_ids, f, indent=2)

    print(f"Ingestion completed! Total IDs stored: {len(all_ids)}")


if __name__ == "__main__":
    ingest_documents()
