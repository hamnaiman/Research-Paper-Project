import os
from dotenv import load_dotenv

load_dotenv()

# Vercel-compatible temporary directories
UPLOAD_DIR = "/tmp/uploads"
CHROMA_DIR = "/tmp/chroma_db"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")

LLM_MODEL_NAME = "openai/gpt-oss-120b"

EMBEDDING_MODEL_NAME = "embed-english-v3.0"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 4

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)