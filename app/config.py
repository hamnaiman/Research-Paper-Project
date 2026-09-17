"""
config.py
---------
All the settings for the app live here so you never have to hunt through
other files to change something (chunk size, model names, folders, etc).
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads variables from a .env file if one exists

# --- Folders -----------------------------------------------------------
UPLOAD_DIR = "data/uploads"      # where uploaded PDFs are saved
CHROMA_DIR = "data/chroma_db"    # where the vector database is persisted

# --- Groq (LLM) ----------------------------------------------------------
# Groq gives a generous free API tier. Get a key at https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL_NAME = "openai/gpt-oss-120b"

# --- Embeddings ----------------------------------------------------------
# This model runs locally on your machine -> completely free, no API key.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between consecutive chunks

# --- Retrieval -------------------------------------------------------------
TOP_K = 4  # how many chunks to fetch from the vector DB per question

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
