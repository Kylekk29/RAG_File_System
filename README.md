**English** | [简体中文](README.zh-CN.md)

# 📚 RAG File System

**Chat with your own documents. Upload files, ask questions, get answers with sources.**

> What if you could drop a pile of books, notes and reports into one app and just *ask* them things? That is what this project does.

---

## 🎯 What Is This?

A small **document Q&A app** built on retrieval-augmented generation (RAG). You upload files, the app splits them into searchable chunks, and when you ask a question it finds the most relevant passages and hands them to an AI model to write the answer.

### Features

| Feature | What It Does |
|---|---|
| **Many file types** | `.txt` `.md` `.pdf` `.docx` `.pptx` `.epub` `.html` `.rtf`, plus source-code files |
| **Multi-book questions** | Ask across up to 5 books at once and merge the results (RRF fusion) |
| **Streaming answers** | Replies appear word by word over Server-Sent Events |
| **Better search** | Parent/child chunking, optional hybrid search and optional cross-encoder re-ranking |
| **Version history** | Re-uploading a file keeps older versions (up to 10 per file) |
| **Chat sessions** | Every conversation is saved, and can be exported |
| **Settings in the UI** | Switch model, temperature and retrieval options without editing code |

---

## 🏗️ How It Works

```
 Upload file ──► Loader ──► Chunks ──► Embeddings ──► FAISS index
                                                         │
 Your question ──► Retrieve top-k ──► (Re-rank) ──► AI model ──► Answer + sources
```

1. **Load:** `document_loader.py` reads the file in whatever format it is.
2. **Chunk:** the text is cut into large parent chunks and small child chunks.
3. **Embed:** `embedding_manager.py` turns chunks into vectors and stores them in a FAISS index.
4. **Ask:** `query_engine.py` finds the best chunks, optionally re-ranks them, and sends them with your question to the model.

---

## 🚀 Getting Started

### What You Need

| Requirement | Details |
|---|---|
| **Python** | 3.10 or newer |
| **API key** | An OpenAI-compatible key. DeepSeek is the default ([platform.deepseek.com](https://platform.deepseek.com)) |
| **Disk space** | A couple of GB for the embedding model |

### Step 1: Get the code

```bash
git clone https://github.com/Kylekk29/RAG_File_System.git
cd RAG_File_System
```

### Step 2: Install

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Configure

Create a `.env` file next to `app.py`:

```ini
API_KEY=your-api-key
BASE_URL=https://api.deepseek.com/v1
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

All other values (chunk sizes, `RETRIEVAL_K`, `RERANK_ENABLED`, `HYBRID_SEARCH_ENABLED`, ...) are listed in `config.py` with defaults.

### Step 4: Run

```bash
python app.py
```

Open **http://127.0.0.1:8000**, upload a file, and start asking.

> ⏱️ The first run downloads the embedding model, so it takes longer. Later runs start in seconds.

---

## 🔌 API

The web page is a thin client over a FastAPI backend.

| Endpoint | Purpose |
|---|---|
| `POST /api/upload` | Upload and index a file |
| `GET /api/books` | List indexed books |
| `POST /api/chat/stream` | Ask one book, streamed |
| `POST /api/chat/multi/stream` | Ask several books, streamed |
| `GET /api/sessions` | List saved chats |
| `GET /api/health` | Health check |

Interactive docs are at `/docs` while the app is running.

## 📁 Project Layout

```
app.py                 FastAPI app and routes
config.py              settings read from .env
document_loader.py     file readers
embedding_manager.py   chunking, embeddings, FAISS
query_engine.py        retrieval, re-ranking, answering
version_manager.py     book version history
session_manager.py     chat sessions
```

Runtime folders (`vector_dbs/`, `uploads/`, `chat_history/`, `logs/`) are created on first run.

## 📄 License

MIT, see [LICENSE](LICENSE).
