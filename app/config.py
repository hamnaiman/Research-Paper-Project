"""
config.py
---------
All settings in one place. GROQ_API_KEY and EMBEDDING_API_KEY are read from
environment variables only — never hardcoded, never committed to GitHub.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Folders -------------------------------------------------------------
UPLOAD_DIR = "data/uploads"
CHROMA_DIR = "data/chroma_db"

# --- Groq (LLM, remote — unchanged) --------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL_NAME = "openai/gpt-oss-120b"

# --- Cohere (Embeddings, remote API — no local model, no PyTorch) --------
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL_NAME = "embed-english-light-v3.0"

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval -------------------------------------------------------------
TOP_K = 4

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)