from fastapi import FastAPI, APIRouter, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import tempfile
import requests
import os

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import google.generativeai as genai

# Constants
TEAM_TOKEN = "f37d7d844f3b77d9dd8e9eb5f95d52fde0ed2fc637e62fdf71ce89eced47df37"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Pydantic Models
class RunRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document")
    questions: List[str] = Field(..., description="List of questions")

class RunResponse(BaseModel):
    answers: List[str]

# Create API Router
router = APIRouter(prefix="/api/v1")

# Utility Functions
def download_pdf(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp.write(r.content)
        temp.flush()
        return temp.name
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Error downloading PDF: {ex}")

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {ex}")
    finally:
        os.remove(pdf_path)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in the provided PDF.")
    return text

def get_text_chunks(text, chunk_size=10000, chunk_overlap=1000):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store, embeddings

def ask_gemini(context, question):
    prompt = f"""
Answer the question as detailed as possible from the provided context, make sure to provide all the details.
If the answer is not in provided context, just say, "answer is not available in the context". Don't provide the wrong answer.

Context:
{context}

Question:
{question}

Answer:
"""
    model = genai.GenerativeModel('gemini-2.5-pro')
    response = model.generate_content(prompt)
    return response.text.strip() if hasattr(response, "text") else str(response)

# Main Endpoint
@router.post("/hackrx/run", response_model=RunResponse)
async def hackrx_run(
    request: Request,
    payload: RunRequest,
    authorization: Optional[str] = Header(None)
):
    # Token validation
    if not authorization or not authorization.startswith("Bearer ") or authorization.split(" ", 1)[1] != TEAM_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token.")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=403, detail="Gemini API key not set on server.")

    # Download and process PDF
    pdf_path = download_pdf(payload.documents)
    pdf_text = extract_text_from_pdf(pdf_path)
    text_chunks = get_text_chunks(pdf_text)
    vector_store, embeddings = create_vector_store(text_chunks)

    # Get answers
    answers = []
    for q in payload.questions:
        docs = vector_store.similarity_search(q, k=4)
        context = "\n\n".join(
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in docs
        )
        answer = ask_gemini(context, q)
        answers.append(answer)

    return RunResponse(answers=answers)

# FastAPI App
app = FastAPI(
    title="Retrieval System API",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Hello from Tesnova"}

@app.post("/api/v1/hackrx/run")
async def hackrx_run():
    return {"message": "HackRX endpoint is working"}

# Include router
app.include_router(router)
