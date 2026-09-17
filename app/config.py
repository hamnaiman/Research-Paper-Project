import os
from dotenv import load_dotenv

load_dotenv()

# --- Folders (Vercel: sirf /tmp likhne layak hai) -------------------------
UPLOAD_DIR = "/tmp/uploads"
CHROMA_DIR = "/tmp/chroma_db"

# --- Groq (LLM, remote) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL_NAME = "openai/gpt-oss-120b"

# --- Cohere (Embeddings, remote API) ---
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL_NAME = "embed-english-light-v3.0"

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval ---
TOP_K = 4

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)