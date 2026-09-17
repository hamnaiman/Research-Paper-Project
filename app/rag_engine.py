
import os
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app import config

# ---------------------------------------------------------------------------
# These are created ONCE when the app starts, and reused for every request.
# ---------------------------------------------------------------------------

_embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

_vectorstore = Chroma(
    collection_name="research_papers",
    embedding_function=_embeddings,
    persist_directory=config.CHROMA_DIR,
)

_llm = ChatGroq(
    model=config.LLM_MODEL_NAME,
    api_key=config.GROQ_API_KEY,
    temperature=0,  # 0 = focused/deterministic answers, good for factual Q&A
)

# The prompt is where we force the "only answer from context" rule.
# The prompt is where we force the "only answer from context" rule.
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a professional Research Paper Assistant. Answer the question
using ONLY the context below, which was extracted from research papers the
user uploaded.

Rules:
- If the answer is not contained in the context, reply with EXACTLY this
  sentence and nothing else: "The answer is not available in the uploaded documents."
- Never use outside knowledge, even if you know the real-world answer.
- Write in clean, natural English. No filler words, no unnecessary commas,
  no repeated phrases.
- For lists (findings, limitations, datasets, steps), use short markdown
  bullet points starting with "- ". One clear idea per bullet.
- For a plain explanation, write 2-4 short sentences. No walls of text.
- Never mention "the context" or "the provided text" in your answer —
  just answer as if you already know the paper.

Context:
{context}

Question: {question}

Answer:"""
)

def get_uploaded_filenames() -> List[str]:
    """Return the list of PDF filenames currently on disk."""
    return sorted(
        f for f in os.listdir(config.UPLOAD_DIR) if f.lower().endswith(".pdf")
    )


def ingest_pdf(file_path: str, filename: str) -> int:
    """
    Load a PDF, split it into overlapping chunks, and store the chunks
    (as embeddings) in ChromaDB. Returns how many chunks were added.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()  # one Document object per PDF page

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    # Tag every chunk with the filename + page number so we can cite the
    # source later when we show the answer to the user.
    for chunk in chunks:
        chunk.metadata["source_file"] = filename
        if "page" in chunk.metadata:
            chunk.metadata["page"] = chunk.metadata["page"] + 1  # 0-index -> human friendly

    if chunks:
        _vectorstore.add_documents(chunks)
        _vectorstore.persist()

    return len(chunks)


def _format_context(docs: List[Document]) -> str:
    """Turn retrieved chunks into one text block the LLM can read, each
    chunk tagged with its source so the model's answer stays traceable."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source: {source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str) -> Dict:
    """
    Full RAG pipeline for one question:
      1. Make sure at least one document has been uploaded.
      2. Retrieve the top-K most relevant chunks.
      3. Ask the LLM to answer using only those chunks.
      4. Return the answer plus a de-duplicated list of sources used.
    """
    if not get_uploaded_filenames():
        return {
            "answer": "Please upload at least one research paper (PDF) before asking questions.",
            "sources": [],
        }

    retriever = _vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})
    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "The answer is not available in the uploaded documents.",
            "sources": [],
        }

    context = _format_context(docs)
    chain = ANSWER_PROMPT | _llm
    response = chain.invoke({"context": context, "question": question})

    sources = [
        {"file": doc.metadata.get("source_file", "unknown"), "page": doc.metadata.get("page", "?")}
        for doc in docs
    ]
    # De-duplicate while preserving order (same page can be retrieved twice)
    seen = set()
    unique_sources = []
    for s in sources:
        key = (s["file"], s["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return {"answer": response.content, "sources": unique_sources}


def reset_knowledge_base():
    """Wipe all stored chunks and delete uploaded files (fresh start)."""
    existing = _vectorstore.get()
    ids = existing.get("ids", [])
    if ids:
        _vectorstore.delete(ids=ids)
    for f in get_uploaded_filenames():
        os.remove(os.path.join(config.UPLOAD_DIR, f))
