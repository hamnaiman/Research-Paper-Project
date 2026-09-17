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
# COHERE EMBEDDINGS
# ============================================================

class CohereAPIEmbeddings(Embeddings):

    def __init__(self, api_key: str, model: str):
        self.client = cohere.Client(api_key)
        self.model = model

    def embed_documents(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        if not texts:
            return []

        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="search_document",
        )

        return response.embeddings

    def embed_query(
        self,
        text: str
    ) -> List[float]:

        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="search_query",
        )

        return response.embeddings[0]


# ============================================================
# LAZY OBJECTS
# ============================================================

_embeddings: Optional[CohereAPIEmbeddings] = None
_vectorstore: Optional[Chroma] = None
_llm: Optional[ChatGroq] = None


# ============================================================
# EMBEDDINGS
# ============================================================

def _get_embeddings():

    global _embeddings

    if _embeddings is None:

        if not config.EMBEDDING_API_KEY:
            raise RuntimeError(
                "EMBEDDING_API_KEY is not configured."
            )

        _embeddings = CohereAPIEmbeddings(
            api_key=config.EMBEDDING_API_KEY,
            model=config.EMBEDDING_MODEL_NAME,
        )

    return _embeddings


# ============================================================
# CHROMA
# ============================================================

def _get_vectorstore():

    global _vectorstore

    if _vectorstore is None:

        _vectorstore = Chroma(
            collection_name="research_papers_v2",
            embedding_function=_get_embeddings(),
            persist_directory=config.CHROMA_DIR,
        )

    return _vectorstore


# ============================================================
# GROQ
# ============================================================

def _get_llm():

    global _llm

    if _llm is None:

        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not configured."
            )

        _llm = ChatGroq(
            model=config.LLM_MODEL_NAME,
            api_key=config.GROQ_API_KEY,
            temperature=0,
        )

    return _llm


# ============================================================
# PROMPT
# ============================================================

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a professional Research Paper Assistant.

Answer the question using ONLY the uploaded research paper
content supplied below.

Rules:

- If the answer is not contained in the uploaded documents,
  reply EXACTLY:

"The answer is not available in the uploaded documents."

- Never use outside knowledge.
- Keep answers concise.
- Use short markdown bullet points for lists.
- For normal explanations, use 2-4 short sentences.
- Never mention "the context" or "provided text".

Context:

{context}

Question:

{question}

Answer:"""
)


# ============================================================
# FILES
# ============================================================

def get_uploaded_filenames():

    if not os.path.exists(config.UPLOAD_DIR):
        return []

    return sorted(
        file
        for file in os.listdir(config.UPLOAD_DIR)
        if file.lower().endswith(".pdf")
    )


# ============================================================
# INGEST PDF
# ============================================================

def ingest_pdf(
    file_path: str,
    filename: str
) -> int:

    if not os.path.exists(file_path):
        raise RuntimeError(
            "Uploaded PDF could not be found."
        )

    loader = PyPDFLoader(file_path)

    pages = loader.load()

    if not pages:
        raise RuntimeError(
            "The PDF contains no readable pages."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(pages)

    if not chunks:
        raise RuntimeError(
            "No readable text was extracted from the PDF."
        )

    for chunk in chunks:

        chunk.metadata["source_file"] = filename

        if "page" in chunk.metadata:

            try:
                chunk.metadata["page"] = (
                    int(chunk.metadata["page"]) + 1
                )
            except Exception:
                pass

    vectorstore = _get_vectorstore()

    vectorstore.add_documents(chunks)

    return len(chunks)


# ============================================================
# FORMAT CONTEXT
# ============================================================

def _format_context(
    docs: List[Document]
) -> str:

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
# ANSWER
# ============================================================

def answer_question(
    question: str
) -> Dict:

    question = question.strip()

    if not question:

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

    context = _format_context(docs)

    chain = ANSWER_PROMPT | _get_llm()

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    sources = []

    for doc in docs:

        source = {
            "file": doc.metadata.get(
                "source_file",
                "unknown"
            ),
            "page": doc.metadata.get(
                "page",
                "?"
            ),
        }

        if source not in sources:
            sources.append(source)

    return {
        "answer": response.content,
        "sources": sources,
    }


# ============================================================
# RESET
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
            f"Failed to reset knowledge base: {exc}"
        )

    if os.path.exists(config.UPLOAD_DIR):

        for filename in os.listdir(
            config.UPLOAD_DIR
        ):

            if filename.lower().endswith(".pdf"):

                path = os.path.join(
                    config.UPLOAD_DIR,
                    filename
                )

                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

    _vectorstore = None