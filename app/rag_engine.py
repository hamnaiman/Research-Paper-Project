"""
rag_engine.py
-------------
The RAG pipeline. Embeddings come from Cohere's Embed API over HTTPS
(no local ML model, no PyTorch, no RAM-heavy startup). We call the official
`cohere` SDK directly through a small wrapper class instead of using the
`langchain_cohere` package, because that package's __init__.py also imports
its ChatCohere code, which breaks on newer `cohere` SDK versions. We only
need embeddings, so this wrapper avoids that broken import chain entirely.

Everything else — chunking, ChromaDB, retrieval, Groq — is unchanged.
"""

import os
from typing import List, Dict, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import cohere

from app import config


class CohereAPIEmbeddings(Embeddings):
    """
    Minimal LangChain-compatible embeddings wrapper around the official
    Cohere Python SDK. Used instead of `langchain_cohere.CohereEmbeddings`
    to avoid that package's broken import chain (see module docstring).
    """

    def __init__(self, api_key: str, model: str):
        self._client = cohere.Client(api_key)
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
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


# ---------------------------------------------------------------------------
# Lazy singletons.
#
# We do NOT create the embeddings client or the Chroma vector store at
# import time with a hard crash if a key is missing. Instead we create them
# on first use, and raise a clear, readable error if the required API key
# isn't set. This keeps FastAPI startup fast and lightweight, and turns a
# confusing stack trace into a message you can actually act on.
# ---------------------------------------------------------------------------

_embeddings: Optional[CohereAPIEmbeddings] = None
_vectorstore: Optional[Chroma] = None
_llm: Optional[ChatGroq] = None


def _get_embeddings() -> CohereAPIEmbeddings:
    global _embeddings
    if _embeddings is None:
        if not config.EMBEDDING_API_KEY:
            raise RuntimeError(
                "Embedding API key is not configured. "
                "Set EMBEDDING_API_KEY in your .env (locally) or in your "
                "Render service's Environment tab (in production)."
            )
        _embeddings = CohereAPIEmbeddings(
            api_key=config.EMBEDDING_API_KEY,
            model=config.EMBEDDING_MODEL_NAME,
        )
    return _embeddings


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name="research_papers_v2",  # _v2: new embedding space,
            embedding_function=_get_embeddings(),  # incompatible with old
            persist_directory=config.CHROMA_DIR,   # local MiniLM vectors
        )
    return _vectorstore


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "Groq API key is not configured. Set GROQ_API_KEY in your "
                "environment."
            )
        _llm = ChatGroq(
            model=config.LLM_MODEL_NAME,
            api_key=config.GROQ_API_KEY,
            temperature=0,
        )
    return _llm


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
    return sorted(
        f for f in os.listdir(config.UPLOAD_DIR) if f.lower().endswith(".pdf")
    )


def ingest_pdf(file_path: str, filename: str) -> int:
    """
    Load a PDF, split it into chunks, embed those chunks via the Cohere API,
    and store them in ChromaDB. If embedding fails (bad key, network error,
    quota), the exception propagates up before anything is written to
    ChromaDB — add_documents() computes all embeddings first and only then
    writes to the collection, so no partial/corrupt vector state is left
    behind. The caller (main.py) is responsible for cleaning up the saved
    file on any failure.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["source_file"] = filename
        if "page" in chunk.metadata:
            chunk.metadata["page"] = chunk.metadata["page"] + 1

    if chunks:
        vectorstore = _get_vectorstore()
        vectorstore.add_documents(chunks)  # embeds via Cohere, then writes
        vectorstore.persist()

    return len(chunks)


def _format_context(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source: {source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str) -> Dict:
    if not get_uploaded_filenames():
        return {
            "answer": "Please upload at least one research paper (PDF) before asking questions.",
            "sources": [],
        }

    retriever = _get_vectorstore().as_retriever(search_kwargs={"k": config.TOP_K})
    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "The answer is not available in the uploaded documents.",
            "sources": [],
        }

    context = _format_context(docs)
    chain = ANSWER_PROMPT | _get_llm()
    response = chain.invoke({"context": context, "question": question})

    sources = [
        {"file": doc.metadata.get("source_file", "unknown"), "page": doc.metadata.get("page", "?")}
        for doc in docs
    ]
    seen = set()
    unique_sources = []
    for s in sources:
        key = (s["file"], s["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return {"answer": response.content, "sources": unique_sources}


def reset_knowledge_base():
    vectorstore = _get_vectorstore()
    existing = vectorstore.get()
    ids = existing.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)
    for f in get_uploaded_filenames():
        os.remove(os.path.join(config.UPLOAD_DIR, f))