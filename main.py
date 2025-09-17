import os
import glob
import pickle
import time
import math
from typing import List, Dict, Any
from dotenv import load_dotenv

import fitz  # PyMuPDF
# import faiss
from pinecone import Pinecone
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


# ---------- Config ----------
DOCUMENTS_DIR = "./documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_DIM = 1536
CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 50
BATCH_SIZE = 32
MAX_RETRIES = 3

load_dotenv()
# API_KEY = os.getenv("OPENAI_API_KEY")
# ---------- Keys ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV", "us-east-1")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "math-tutor-index")


if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is required")

client = OpenAI(api_key=OPENAI_API_KEY)
pinecone = Pinecone(api_key=PINECONE_API_KEY)

app = FastAPI(
    title="SkuleIQ API",
    description="""
        Math tutor API
    """,
    version="1.0.0",
    contact={
        "name": "SkuleIQ",
        "url": "https://skuleiq.com/",
        "email": "hello@skuleiq.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return {"message": "SkuleIQ API"}



# Create or connect to index
if INDEX_NAME not in [i["name"] for i in pinecone.list_indexes()]:
    pinecone.create_index(name=INDEX_NAME, dimension=EMBEDDING_DIM, metric="cosine")
index = pinecone.Index(INDEX_NAME)

# Local store of chunks metadata (IDs -> text)
chunks_store: List[str] = []

# ---------- Utilities ----------

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    texts = [page.get_text() for page in doc]
    return "\n\n".join(texts)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_embeddings_batch(texts: List[str], batch_size: int = BATCH_SIZE, max_retries: int = MAX_RETRIES):
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                batch_emb = [item.embedding for item in resp.data]
                all_emb.extend(batch_emb)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e
    return np.array(all_emb, dtype="float32")


# ---------- Ingest ----------

def ingest_documents():
    global chunks_store

    for fname in os.listdir(DOCUMENTS_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(DOCUMENTS_DIR, fname)
        try:
            text = extract_text_from_pdf(path)
        except Exception:
            continue
        chunks = chunk_text(text)
        if not chunks:
            continue

        embeddings = get_embeddings_batch(chunks)

        # Upsert to Pinecone
        vectors = [(f"chunk-{len(chunks_store)+i}", emb.tolist(), {"text": chunk}) for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
        index.upsert(vectors=vectors)
        chunks_store.extend(chunks)

    return {"status": "ingested", "total_chunks": len(chunks_store)}


# @app.on_event("startup")
# def startup_event():
#     ingest_documents()


# ---------- Upload endpoint ----------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    save_path = os.path.join(DOCUMENTS_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    # Ingest new doc
    try:
        text = extract_text_from_pdf(save_path)
        chunks = chunk_text(text)
        if chunks:
            embeddings = get_embeddings_batch(chunks)
            vectors = [(f"chunk-{len(chunks_store)+i}", emb.tolist(), {"text": chunk}) for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
            index.upsert(vectors=vectors)
            chunks_store.extend(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest uploaded PDF: {e}")

    return {"status": "uploaded_and_ingested", "filename": file.filename}


# ---------- Ask endpoint ----------

class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    temperature: float = 0.2


@app.post("/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Embed question
    q_resp = client.embeddings.create(model=EMBEDDING_MODEL, input=question)
    q_emb = q_resp.data[0].embedding

    # Query Pinecone
    results = index.query(vector=q_emb, top_k=req.top_k, include_metadata=True)
    retrieved = [match["metadata"]["text"] for match in results["matches"]]

    context = "\n\n".join(retrieved)

    # system_msg = "You are an expert math tutor. Use the provided context to answer the question with clear step-by-step explanation and final answer."
    system_msg = """
    You are an expert math tutor who can handle both problem-solving and concept explanations. Your role is to:

    FOR PROBLEM-SOLVING QUESTIONS (like 'Solve 2x + 4 = 24'):
    1. NEVER give the complete solution at once - break problems into logical steps
    2. Ask questions to guide the student's thinking at each step
    3. Wait for the student's response before moving forward
    4. Confirm correctness and explain why each step works
    5. Encourage the student to verify their answer by substitution at the end
    6. Explain the purpose of each mathematical operation

    FOR CONCEPT EXPLANATION QUESTIONS (like 'Explain trigonometry'):
    1. Provide clear, comprehensive explanations with practical examples
    2. Use simple language and relatable analogies
    3. Include step-by-step examples to illustrate concepts
    4. Ask if the student needs clarification on any part
    5. Offer to work through specific problems related to the concept

    GENERAL RULES:
    - Keep a friendly, encouraging tone while being professional
    - Use proper markdown formatting for better readability
    - If you're not sure what the student wants, ask for clarification
    - Keep responses conversational and focused on the student's needs
    - When responding to LaTeX mathematical expressions (wrapped in $$), acknowledge the mathematical notation and work with it appropriately.

    """

    user_msg = f"Question: {question}\n\nContext:\n{context}\n\nPlease answer with step-by-step reasoning."

    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=req.temperature,
            max_tokens=1024
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return {"answer": answer}


# ---------- Health ----------

@app.get("/health")
def health():
    return {"status": "ok", "index": INDEX_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
