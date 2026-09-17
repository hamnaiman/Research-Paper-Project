"""
rag_engine.py
-------------
RAG pipeline using:

PDF
→ Text extraction
→ Chunking
→ Cohere API embeddings
→ ChromaDB
→ Similarity retrieval
→ Groq LLM

Cohere embeddings are accessed through the official Cohere SDK.
No HuggingFace, SentenceTransformers, PyTorch, or local ML model is used.
This keeps the application lightweight for Render's memory limit.
"""

import os
from typing import List, Dict, Optional

import cohere

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app import config


# ============================================================
# COHERE EMBEDDING WRAPPER
# ============================================================

class CohereAPIEmbeddings(Embeddings):
    """
    Lightweight LangChain-compatible wrapper around Cohere's
    official embedding API.

    No local embedding model is loaded.
    """

    def __init__(self, api_key: str, model: str):
        self._client = cohere.Client(api_key)
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        response = self._client.embed(
            texts=texts,
            model=self._model,
            input_type="search_document",
        )

        return response.embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self._client.embed(
            texts=[text],
            model=self._model,
            input_type="search_query",
        )

        return response.embeddings[0]


# ============================================================
# LAZY SINGLETONS
# ============================================================

_embeddings: Optional[CohereAPIEmbeddings] = None
_vectorstore: Optional[Chroma] = None
_llm: Optional[ChatGroq] = None


# ============================================================
# EMBEDDINGS
# ============================================================

def _get_embeddings() -> CohereAPIEmbeddings:
    global _embeddings

    if _embeddings is None:

        if not config.EMBEDDING_API_KEY:
            raise RuntimeError(
                "Embedding API key is not configured. "
                "Set EMBEDDING_API_KEY in your .env file locally "
                "or in Render Environment Variables."
            )

        _embeddings = CohereAPIEmbeddings(
            api_key=config.EMBEDDING_API_KEY,
            model=config.EMBEDDING_MODEL_NAME,
        )

    return _embeddings


# ============================================================
# CHROMA VECTOR STORE
# ============================================================

def _get_vectorstore() -> Chroma:
    global _vectorstore

    if _vectorstore is None:

        _vectorstore = Chroma(
            collection_name="research_papers_v2",
            embedding_function=_get_embeddings(),
            persist_directory=config.CHROMA_DIR,
        )

    return _vectorstore


# ============================================================
# GROQ LLM
# ============================================================

def _get_llm() -> ChatGroq:
    global _llm

    if _llm is None:

        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "Groq API key is not configured. "
                "Set GROQ_API_KEY in your environment variables."
            )

        _llm = ChatGroq(
            model=config.LLM_MODEL_NAME,
            api_key=config.GROQ_API_KEY,
            temperature=0,
        )

    return _llm


# ============================================================
# ANSWER PROMPT
# ============================================================

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
- Never mention "the context" or "the provided text" in your answer.
- Just answer as if you already know the paper.

Context:
{context}

Question:
{question}

Answer:"""
)


# ============================================================
# GET UPLOADED PDF FILES
# ============================================================

def get_uploaded_filenames() -> List[str]:
    if not os.path.exists(config.UPLOAD_DIR):
        return []

    return sorted(
        f
        for f in os.listdir(config.UPLOAD_DIR)
        if f.lower().endswith(".pdf")
    )


# ============================================================
# INGEST PDF
# ============================================================

def ingest_pdf(file_path: str, filename: str) -> int:
    """
    Load PDF, split into chunks, create Cohere embeddings,
    and store the vectors in ChromaDB.
    """

    # ----------------------------
    # Load PDF
    # ----------------------------

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    if not pages:
        raise RuntimeError("The PDF contains no readable pages.")

    # ----------------------------
    # Split into chunks
    # ----------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(pages)

    if not chunks:
        raise RuntimeError(
            "No readable text could be extracted from the PDF."
        )

    # ----------------------------
    # Add metadata
    # ----------------------------

    for chunk in chunks:

        chunk.metadata["source_file"] = filename

        if "page" in chunk.metadata:
            try:
                chunk.metadata["page"] = (
                    int(chunk.metadata["page"]) + 1
                )
            except (TypeError, ValueError):
                pass

    # ----------------------------
    # Store in ChromaDB
    # ----------------------------

    vectorstore = _get_vectorstore()

    vectorstore.add_documents(chunks)

    # No vectorstore.persist() here.
    # Modern langchain-chroma handles persistence automatically
    # through persist_directory.

    return len(chunks)


# ============================================================
# FORMAT RETRIEVED CONTEXT
# ============================================================

def _format_context(docs: List[Document]) -> str:

    parts = []

    for doc in docs:

        source = doc.metadata.get(
            "source_file",
            "unknown"
        )

        page = doc.metadata.get(
            "page",
            "?"
        )

        parts.append(
            f"[Source: {source}, page {page}]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)


# ============================================================
# ASK QUESTION
# ============================================================

def answer_question(question: str) -> Dict:

    if not question or not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    if not get_uploaded_filenames():
        return {
            "answer": (
                "Please upload at least one research paper "
                "(PDF) before asking questions."
            ),
            "sources": [],
        }

    # ----------------------------
    # Retrieve relevant chunks
    # ----------------------------

    vectorstore = _get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": config.TOP_K
        }
    )

    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": (
                "The answer is not available in the "
                "uploaded documents."
            ),
            "sources": [],
        }

    # ----------------------------
    # Build context
    # ----------------------------

    context = _format_context(docs)

    # ----------------------------
    # Ask Groq
    # ----------------------------

    chain = ANSWER_PROMPT | _get_llm()

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    # ----------------------------
    # Build sources
    # ----------------------------

    sources = []

    for doc in docs:

        sources.append(
            {
                "file": doc.metadata.get(
                    "source_file",
                    "unknown"
                ),
                "page": doc.metadata.get(
                    "page",
                    "?"
                ),
            }
        )

    # ----------------------------
    # Remove duplicate sources
    # ----------------------------

    seen = set()
    unique_sources = []

    for source in sources:

        key = (
            source["file"],
            source["page"],
        )

        if key not in seen:

            seen.add(key)
            unique_sources.append(source)

    return {
        "answer": response.content,
        "sources": unique_sources,
    }


# ============================================================
# RESET KNOWLEDGE BASE
# ============================================================

def reset_knowledge_base():

    global _vectorstore

    vectorstore = _get_vectorstore()

    try:

        existing = vectorstore.get()

        ids = existing.get(
            "ids",
            []
        )

        if ids:
            vectorstore.delete(
                ids=ids
            )

    except Exception as exc:

        raise RuntimeError(
            f"Failed to reset ChromaDB: {exc}"
        ) from exc

    # ----------------------------
    # Delete uploaded PDFs
    # ----------------------------

    for filename in get_uploaded_filenames():

        file_path = os.path.join(
            config.UPLOAD_DIR,
            filename
        )

        try:
            os.remove(file_path)

        except FileNotFoundError:
            pass

    # Force Chroma to be recreated
    # on the next request.

    _vectorstore = None