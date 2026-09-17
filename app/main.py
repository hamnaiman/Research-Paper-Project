
import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import config
from app import rag_engine

app = FastAPI(title="Research Paper Assistant (RAG)")

# Allow the frontend (or any tool like Postman) to call this API freely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


@app.post("/upload")
async def upload_paper(file: UploadFile = File(...)):
    """Upload a single PDF research paper and index it into the vector DB."""

    # --- Graceful handling: unsupported file format -----------------
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a PDF file (.pdf).",
        )

    dest_path = os.path.join(config.UPLOAD_DIR, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        num_chunks = rag_engine.ingest_pdf(dest_path, file.filename)
    except Exception as e:
        os.remove(dest_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    # --- Graceful handling: PDF had no extractable text (e.g. scanned) ---
    if num_chunks == 0:
        os.remove(dest_path)
        raise HTTPException(
            status_code=400,
            detail="No readable text found in this PDF (it may be a scanned image without OCR).",
        )

    return {
        "message": f"'{file.filename}' uploaded and indexed successfully.",
        "chunks_added": num_chunks,
    }


@app.post("/ask")
async def ask_question(payload: Question):
    """Ask a question about the uploaded papers."""

    # --- Graceful handling: empty question ---------------------------
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = rag_engine.answer_question(question)
    return result


@app.get("/documents")
async def list_documents():
    """List every PDF currently indexed."""
    return {"documents": rag_engine.get_uploaded_filenames()}


@app.post("/reset")
async def reset():
    """Delete all uploaded papers and clear the vector database."""
    rag_engine.reset_knowledge_base()
    return {"message": "Knowledge base cleared."}


# --- Serve the simple chat frontend -----------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")
