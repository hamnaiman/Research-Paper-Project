# 📄 Research Paper Assistant (RAG)

A beginner-friendly app that lets you upload research papers (PDF) and ask
questions about them. It answers **only** from the content you uploaded —
if the answer isn't in your papers, it tells you so instead of guessing.

Built with: **FastAPI** + **LangChain** + **ChromaDB** + **Groq (LLM)**.

---

## How it works (in plain English)

This is a **RAG** app — Retrieval-Augmented Generation. Instead of asking
an AI to answer purely from what it memorized during training (which can
be outdated or made up), we:

1. **Split** each uploaded PDF into small overlapping text chunks (~1000
   characters each).
2. **Embed** each chunk — turn it into a list of numbers ("embedding")
   that represents its meaning — using a free local model
   (`all-MiniLM-L6-v2`).
3. **Store** those embeddings in **ChromaDB**, a vector database that can
   quickly find "chunks that mean something similar to X".
4. When you ask a question, we **retrieve** the top 4 most relevant chunks
   from ChromaDB.
5. We hand those chunks + your question to an **LLM** (Groq's
   `llama-3.1-8b-instant`, free tier) with a strict instruction: *"answer
   using only this context; if it's not here, say so."*
6. We show you the answer **plus** which file/page it came from.

```
PDF --> split into chunks --> embed chunks --> store in ChromaDB
                                                      |
question --> embed question --> search ChromaDB  <---
                                       |
                          top matching chunks
                                       |
                          LLM (answers using only these chunks)
                                       |
                             answer + sources shown to user
```

---

## Project structure

```
research-paper-assistant/
├── app/
│   ├── config.py       # all settings (chunk size, model names, folders)
│   ├── rag_engine.py    # the RAG logic: ingest, retrieve, answer
│   └── main.py           # FastAPI endpoints
├── static/
│   └── index.html          # simple chat UI (no framework needed)
├── data/
│   ├── uploads/               # uploaded PDFs are saved here
│   └── chroma_db/              # the vector database lives here
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup (step by step)

### 1. Get a free Groq API key
Go to https://console.groq.com/keys, sign up (free), and create an API key.
Groq's free tier is generous and fast — no credit card needed to start.

### 2. Clone / download this project and open a terminal in it

### 3. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```
> The first run will also download the small local embedding model
> (~90MB) — this only happens once.

### 5. Add your API key
```bash
cp .env.example .env
```
Open `.env` and paste your Groq key:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

### 6. Run the app
```bash
uvicorn app.main:app --reload
```

### 7. Open the app
Go to **http://127.0.0.1:8000** in your browser. Upload a PDF, wait for it
to say "indexed successfully", then start asking questions.

---

## API endpoints (if you want to test with curl / Postman instead of the UI)

| Method | Endpoint      | Description                          |
|--------|---------------|---------------------------------------|
| POST   | `/upload`     | Upload one PDF (form field: `file`)   |
| POST   | `/ask`        | `{"question": "..."}` -> answer       |
| GET    | `/documents`  | List uploaded PDFs                    |
| POST   | `/reset`      | Clear all papers and the vector DB    |

Example:
```bash
curl -X POST -F "file=@paper.pdf" http://127.0.0.1:8000/upload

curl -X POST http://127.0.0.1:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What methodology is used?"}'
```

---

## Example questions to try
- Summarize this paper.
- What problem does this paper solve?
- What methodology is used?
- What datasets and algorithms are used?
- What are the key findings?
- What limitations and future work are mentioned?

---

## Edge cases handled

| Situation                          | Behavior                                                        |
|-------------------------------------|-------------------------------------------------------------------|
| No document uploaded, question asked | Returns a message asking the user to upload a paper first        |
| Empty question submitted             | Returns `400 Bad Request: "Question cannot be empty."`             |
| Non-PDF file uploaded                | Returns `400 Bad Request: "Unsupported file format..."`            |
| Scanned PDF with no extractable text | Returns `400 Bad Request` explaining no readable text was found  |
| Question not answerable from papers  | LLM replies: "The answer is not available in the uploaded documents." |

---

## Swapping the LLM provider (optional)

The assignment allows Gemini, Groq, or OpenAI. This project defaults to
**Groq** because it's free and fast. To switch to OpenAI instead, in
`app/rag_engine.py` replace:
```python
from langchain_groq import ChatGroq
_llm = ChatGroq(model=config.LLM_MODEL_NAME, api_key=config.GROQ_API_KEY, temperature=0)
```
with:
```python
from langchain_openai import ChatOpenAI
_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
```
(and `pip install langchain-openai`).

---

## Submitting to GitHub

```bash
git init
git add .
git commit -m "Research Paper Assistant (RAG) - FastAPI + LangChain + ChromaDB + Groq"
git branch -M main
git remote add origin <your-empty-github-repo-url>
git push -u origin main
```

Make sure `.env` is **not** committed (it's already in `.gitignore`) so
your API key stays private — submit `.env.example` instead.
