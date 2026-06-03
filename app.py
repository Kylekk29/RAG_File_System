"""
app.py — FastAPI application with full commercial-grade feature set:
  • Settings CRUD  (/api/settings, /api/providers)
  • Streaming SSE  (/api/chat/stream)
  • Chat history   (/api/chat/history/…)
  • Book management (upload, list, delete, version history)
  • Health check   (/api/health)
"""

import glob
import json
import logging
import os
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

import uvicorn

import config
import document_loader
from chat_history_manager import ChatHistoryManager
from embedding_manager import BookEmbeddingManager
from query_engine import BookQueryEngine
from session_manager import SessionManager
from settings_manager import SettingsManager
from version_manager import BookVersionManager

logger = logging.getLogger(__name__)

# ── App bootstrap ──────────────────────────────────────────────────────────────
app = FastAPI(title="RAG — Document Q&A", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

settings_manager = SettingsManager()
embedding_manager = BookEmbeddingManager()
query_engine = BookQueryEngine(settings_manager=settings_manager, embedding_manager=embedding_manager)
version_manager = BookVersionManager()
chat_history_manager = ChatHistoryManager()
session_manager = SessionManager()

current_active_book: str = ""
current_active_books: List[str] = []
current_session_id: str = ""

# ── Pydantic models ────────────────────────────────────────────────────────────

class SettingsUpdate(BaseModel):
    model: Optional[Dict[str, Any]] = None
    rag: Optional[Dict[str, Any]] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_book_index_dir(book_name: str) -> Optional[str]:
    book_id = BookVersionManager.clean_id(book_name)
    db_id = version_manager.get_current_db_id(book_id)
    if db_id:
        idx_dir = os.path.join(config.VECTOR_DB_DIR, db_id)
        if os.path.exists(os.path.join(idx_dir, "index.faiss")):
            return idx_dir
    for meta_file in glob.glob(os.path.join(config.VECTOR_DB_DIR, "*", "book_meta.json")):
        try:
            with open(meta_file, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("book_name") == book_name:
                idx_dir = os.path.dirname(meta_file)
                if os.path.exists(os.path.join(idx_dir, "index.faiss")):
                    return idx_dir
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _book_index_exists(book_name: str) -> bool:
    return _find_book_index_dir(book_name) is not None


def _load_book_indices(book_names: List[str]) -> List:
    """Load FAISS indices for a list of book names. Returns list of (name, index) tuples."""
    indices = []
    errors = []
    for name in book_names:
        try:
            idx = embedding_manager.get_vector_index(name)
            indices.append((name, idx))
            logger.debug("Loaded index for book: %s", name)
        except FileNotFoundError:
            errors.append(name)
    if errors:
        logger.warning("Could not load indices for: %s", errors)
    logger.debug("Loaded %d/%d book indices", len(indices), len(book_names))
    return indices


# ── Static ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(config.BASE_DIR, "index.html")
    with open(index_path, encoding="utf-8") as fh:
        return fh.read()


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    logger.debug("Health check")
    return {
        "status": "ok",
        "version": "2.0.0",
        "timestamp": time.time(),
        "active_book": current_active_book,
        "active_books": current_active_books,
    }


# ── Settings ───────────────────────────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings():
    logger.debug("GET /api/settings")
    s = settings_manager.get_settings()
    # Never expose api_key over the wire (show masked hint only)
    model_s = dict(s.get("model", {}))
    model_s.pop("api_key", None)
    s["model"] = model_s
    return {"success": True, "settings": s}


@app.post("/api/settings")
async def update_settings(body: SettingsUpdate):
    updates: Dict[str, Any] = {}
    if body.model:
        updates["model"] = body.model
    if body.rag:
        updates["rag"] = body.rag
    if not updates:
        raise HTTPException(status_code=400, detail="No settings provided")
    logger.info("Updating settings: model=%s, rag=%s", bool(body.model), bool(body.rag))

    # Rebuild model_provider LLM config when model settings change
    if "model" in updates:
        query_engine.model_provider.settings_mgr.update_settings(updates)
    else:
        settings_manager.update_settings(updates)

    new_settings = settings_manager.get_settings()
    model_s = dict(new_settings.get("model", {}))
    model_s.pop("api_key", None)
    new_settings["model"] = model_s
    return {"success": True, "settings": new_settings}


@app.get("/api/providers")
async def list_providers():
    presets = SettingsManager.get_providers()
    providers = [{"id": k, "name": v.get("label", k)} for k, v in presets.items()]
    return {"success": True, "providers": providers}


# ── Books ──────────────────────────────────────────────────────────────────────

@app.get("/api/books")
async def list_books():
    logger.info("GET /api/books")
    books = embedding_manager.list_books()
    # Enrich with history stats
    for b in books:
        try:
            stats = chat_history_manager.get_stats(b.get("name", ""))
            b["history_exchanges"] = stats["exchanges"]
        except Exception:
            b["history_exchanges"] = 0
    logger.info("Listed %d books", len(books))
    return {"books": books, "current": current_active_book, "active_books": current_active_books}


@app.post("/api/upload")
async def upload_book(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
):
    global current_active_book, current_active_books

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename)[-1].lower()
    supported = document_loader.get_supported_extensions()
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(supported)}",
        )

    book_name = os.path.splitext(file.filename)[0]
    temp_path = os.path.join(config.UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    logger.info("Uploading file: filename=%s, book_name=%s, ext=%s, size=%d",
                 file.filename, book_name, ext, file.size or 0)
    try:
        content = await file.read()
        with open(temp_path, "wb") as fh:
            fh.write(content)
        logger.debug("Saved temp file: %s (%d bytes)", temp_path, len(content))

        result = embedding_manager.ingest_book(
            temp_path, book_name=book_name, source_filename=file.filename,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        current_active_book = result["book_name"]
        current_active_books = [result["book_name"]]

        resp: Dict[str, Any] = {
            "success": True,
            "book_name": result["book_name"],
            "action": result.get("action", "created"),
            "version": result.get("version_number", 1),
            "message": result.get("message", ""),
            "file_type": result.get("file_type", ext),
        }
        if "chunk_count" in result:
            resp["chunk_count"] = result["chunk_count"]
        if "warning" in result:
            resp["warning"] = result["warning"]
        return resp

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Upload failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/api/books/{book_name}")
async def get_book_details(book_name: str):
    logger.info("GET /api/books/%s", book_name)
    try:
        book_info = embedding_manager.get_book_info(book_name)
        book_id = BookVersionManager.clean_id(book_name)
        book_info["version_history"] = version_manager.get_version_history(
            book_id, book_name=book_name
        )
        book_info["history_stats"] = chat_history_manager.get_stats(book_name)
        return {"success": True, "book": book_info}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/books/{book_name}/history_versions")
async def get_book_version_history(book_name: str):
    book_id = BookVersionManager.clean_id(book_name)
    versions = version_manager.get_version_history(book_id, book_name=book_name)
    return {"success": True, "book_name": book_name, "versions": versions}


@app.post("/api/set_active_book")
async def set_active_book(book_name: str = Form(...)):
    global current_active_book, current_active_books
    logger.info("Setting active book: %s", book_name)
    if not _book_index_exists(book_name):
        logger.warning("Book not found: %s", book_name)
        raise HTTPException(status_code=404, detail=f"Book '{book_name}' not found")
    current_active_book = book_name
    current_active_books = [book_name]
    history = chat_history_manager.load_history(book_name)
    logger.info("Active book set: %s (history: %d messages)", book_name, len(history))
    return {"success": True, "book_name": book_name, "history": history}


@app.post("/api/set_active_books")
async def set_active_books(book_names: str = Form(...)):
    global current_active_books, current_active_book
    names = [n.strip() for n in book_names.split(",") if n.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="At least one book name required")
    missing = [n for n in names if not _book_index_exists(n)]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Books not found: {', '.join(missing)}",
        )
    current_active_books = names
    current_active_book = names[0]  # keep single for backward compat
    return {"success": True, "book_names": names}


@app.delete("/api/books/{book_name}")
async def delete_book(book_name: str):
    global current_active_book, current_active_books
    logger.info("DELETE /api/books/%s", book_name)
    if embedding_manager.delete_book(book_name):
        if current_active_book == book_name:
            current_active_book = ""
            logger.debug("Cleared active book (was '%s')", book_name)
        current_active_books = [b for b in current_active_books if b != book_name]
        chat_history_manager.clear_history(book_name)
        logger.info("Book deleted successfully: '%s'", book_name)
        return {"success": True, "book_name": book_name}
    logger.warning("Book not found for deletion: '%s'", book_name)
    raise HTTPException(status_code=404, detail=f"Book '{book_name}' not found")


# ── Sessions ────────────────────────────────────────────────────────────────────

class CreateSessionBody(BaseModel):
    name: str
    selected_books: Optional[List[str]] = None


class UpdateSessionBody(BaseModel):
    name: Optional[str] = None
    selected_books: Optional[List[str]] = None
    active_book: Optional[str] = None


@app.get("/api/sessions")
async def list_sessions():
    logger.debug("GET /api/sessions")
    sessions = session_manager.list_sessions()
    logger.debug("Listed %d sessions", len(sessions))
    return {"success": True, "sessions": sessions}


@app.post("/api/sessions")
async def create_session(body: CreateSessionBody):
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Session name is required")
    logger.info("Creating session: name=%s, selected_books=%s", body.name, body.selected_books)
    session = session_manager.create_session(body.name.strip(), body.selected_books)
    logger.info("Session created: id=%s", session.get("id"))
    return {"success": True, "session": session}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    logger.debug("GET /api/sessions/%s", session_id)
    session = session_manager.get_session(session_id)
    if not session:
        logger.warning("Session not found: %s", session_id)
        raise HTTPException(status_code=404, detail="Session not found")
    messages = session_manager.load_messages(session_id)
    logger.debug("Session %s: %d messages", session_id, len(messages))
    return {"success": True, "session": session, "messages": messages}


@app.patch("/api/sessions/{session_id}")
async def update_session(session_id: str, body: UpdateSessionBody):
    updates = {}
    if body.name is not None:
        updates["name"] = body.name
    if body.selected_books is not None:
        updates["selected_books"] = body.selected_books
    if body.active_book is not None:
        updates["active_book"] = body.active_book
    logger.info("Updating session %s: %s", session_id, updates)
    session = session_manager.update_session(session_id, updates)
    if not session:
        logger.warning("Session not found for update: %s", session_id)
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "session": session}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    global current_session_id
    logger.info("Deleting session: %s", session_id)
    if session_manager.delete_session(session_id):
        if current_session_id == session_id:
            current_session_id = ""
            logger.debug("Cleared current_session_id")
        logger.info("Session deleted: %s", session_id)
        return {"success": True}
    logger.warning("Session not found for deletion: %s", session_id)
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    logger.debug("GET /api/sessions/%s/messages", session_id)
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = session_manager.load_messages(session_id)
    stats = session_manager.get_stats(session_id)
    logger.debug("Session %s: %d messages", session_id, len(messages))
    return {"success": True, "messages": messages, "stats": stats}


@app.delete("/api/sessions/{session_id}/messages")
async def clear_session_messages(session_id: str):
    logger.info("Clearing messages for session: %s", session_id)
    session_manager.clear_messages(session_id)
    return {"success": True}


@app.get("/api/sessions/{session_id}/export")
async def export_session_messages(session_id: str, format: str = "json"):
    if format not in ("json", "markdown"):
        raise HTTPException(status_code=400, detail="format must be 'json' or 'markdown'")
    logger.info("Exporting session %s as %s", session_id, format)
    content = session_manager.export_messages(session_id, fmt=format)
    media = "application/json" if format == "json" else "text/markdown"
    filename = f"session_{session_id}.{format if format == 'json' else 'md'}"
    return StreamingResponse(
        iter([content]),
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Chat: blocking ─────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_with_book(
    session_id: Optional[str] = Form(None),
    book_name: Optional[str] = Form(None),
    question: str = Form(...),
    prompt_type: str = Form("book_qa"),
    retrieval_k: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
    mode: str = Form("rag"),
):
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    logger.info("POST /api/chat: book=%s, prompt_type=%s, retrieval_k=%s, mode=%s, query=%.60s",
                 book_name, prompt_type, retrieval_k, mode, question)

    # Resolve book_name from session if needed
    if not book_name and session_id:
        session = session_manager.get_session(session_id)
        if session:
            sel = session.get("selected_books") or []
            book_name = sel[0] if sel else None
            logger.debug("Resolved book_name from session %s: %s", session_id, book_name)

    if not book_name:
        raise HTTPException(status_code=400, detail="book_name or session_id with selected_books is required")

    try:
        runtime_params: Dict[str, Any] = {}
        if temperature is not None:
            runtime_params["temperature"] = temperature
        if top_p is not None:
            runtime_params["top_p"] = top_p
        if runtime_params:
            logger.debug("Runtime params: %s", runtime_params)

        kw = dict(book_name=book_name, query=question, prompt_type=prompt_type,
                   retrieval_k=retrieval_k, runtime_params=runtime_params or None, mode=mode)
        if mode != "full":
            kw["vector_index"] = embedding_manager.get_vector_index(book_name)
        result = query_engine.ask_question(**kw)
        logger.info("Chat response: %d chunks, %d sources, response_len=%d",
                     result["retrieved_chunks"], len(result["sources"]), len(result["response"]))

        # Persist to session or legacy history
        if session_id:
            session_manager.add_exchange(
                session_id, question, result["response"],
                metadata={
                    "chunks": result["retrieved_chunks"],
                    "prompt_type": prompt_type,
                    "sources": [s["content_preview"] for s in result["sources"][:3]],
                    "book_name": book_name,
                },
            )
        else:
            chat_history_manager.add_exchange(
                book_name, question, result["response"],
                metadata={
                    "chunks": result["retrieved_chunks"],
                    "prompt_type": prompt_type,
                    "sources": [s["content_preview"] for s in result["sources"][:3]],
                },
            )
        return {
            "success": True,
            "response": result["response"],
            "retrieved_chunks": result["retrieved_chunks"],
            "sources": result["sources"],
            "book_name": book_name,
        }
    except FileNotFoundError as exc:
        logger.warning("Chat failed - book not found: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Chat failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Chat: streaming SSE ────────────────────────────────────────────────────────

@app.post("/api/chat/stream")
async def stream_chat(
    session_id: Optional[str] = Form(None),
    book_name: Optional[str] = Form(None),
    question: str = Form(...),
    prompt_type: str = Form("book_qa"),
    retrieval_k: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
    mode: str = Form("rag"),
):
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    if not book_name and session_id:
        session = session_manager.get_session(session_id)
        if session:
            sel = session.get("selected_books") or []
            book_name = sel[0] if sel else None

    if not book_name:
        raise HTTPException(status_code=400, detail="book_name or session_id with selected_books is required")

    logger.info("POST /api/chat/stream: book=%s, prompt_type=%s, mode=%s, query=%.60s", book_name, prompt_type, mode, question)

    runtime_params: Dict[str, Any] = {}
    if temperature is not None:
        runtime_params["temperature"] = temperature
    if top_p is not None:
        runtime_params["top_p"] = top_p

    async def event_generator():
        full_response = ""
        sources = []
        if mode == "full":
            for event in query_engine.stream_question(
                book_name, question, vector_index=None,
                prompt_type=prompt_type, retrieval_k=retrieval_k,
                runtime_params=runtime_params or None, mode=mode,
            ):
                if event["type"] == "meta":
                    sources = event.get("sources", [])
                elif event["type"] == "chunk":
                    full_response += event.get("text", "")
                payload = json.dumps(event, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            if full_response:
                meta = {"chunks": len(sources), "prompt_type": prompt_type, "sources": [s["content_preview"] for s in sources[:3]], "mode": mode}
                if session_id:
                    session_manager.add_exchange(session_id, question, full_response, metadata={**meta, "book_name": book_name})
                else:
                    chat_history_manager.add_exchange(book_name, question, full_response, metadata=meta)
            return

        try:
            vector_index = embedding_manager.get_vector_index(book_name)
            logger.debug("Stream: FAISS index loaded for %s", book_name)
        except FileNotFoundError as exc:
            payload = json.dumps({"type": "error", "message": str(exc), "code": 404})
            yield f"data: {payload}\n\n"
            return
        except Exception as exc:
            payload = json.dumps({"type": "error", "message": str(exc), "code": 500})
            yield f"data: {payload}\n\n"
            return

        event_count = 0
        for event in query_engine.stream_question(
            book_name,
            question,
            vector_index=vector_index,
            prompt_type=prompt_type,
            retrieval_k=retrieval_k,
            runtime_params=runtime_params or None,
            mode=mode,
        ):
            if event["type"] == "meta":
                sources = event.get("sources", [])
                logger.debug("Stream: received %d sources", len(sources))
            elif event["type"] == "chunk":
                full_response += event.get("text", "")
            payload = json.dumps(event, ensure_ascii=False)
            yield f"data: {payload}\n\n"
            event_count += 1

        logger.info("Stream complete: book=%s, events=%d, response_len=%d", book_name, event_count, len(full_response))

        if full_response:
            meta = {
                "chunks": len(sources),
                "prompt_type": prompt_type,
                "sources": [s["content_preview"] for s in sources[:3]],
            }
            if session_id:
                session_manager.add_exchange(session_id, question, full_response, metadata={**meta, "book_name": book_name})
            else:
                chat_history_manager.add_exchange(book_name, question, full_response, metadata=meta)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Chat: multi-book blocking ─────────────────────────────────────────────────

@app.post("/api/chat/multi")
async def chat_with_multiple_books(
    session_id: Optional[str] = Form(None),
    book_names: Optional[str] = Form(None),
    question: str = Form(...),
    prompt_type: str = Form("book_qa"),
    retrieval_k: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
):
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    # Resolve book_names from session if not provided
    if not book_names and session_id:
        session = session_manager.get_session(session_id)
        if session:
            sel = session.get("selected_books") or []
            book_names = ",".join(sel) if sel else None

    if not book_names:
        raise HTTPException(status_code=400, detail="book_names or session_id with selected_books is required")

    names = [n.strip() for n in book_names.split(",") if n.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="At least one book name required")

    logger.info("POST /api/chat/multi: books=%s, prompt_type=%s, query=%.60s", names, prompt_type, question)

    runtime_params: Dict[str, Any] = {}
    if temperature is not None:
        runtime_params["temperature"] = temperature
    if top_p is not None:
        runtime_params["top_p"] = top_p

    try:
        indices = _load_book_indices(names)
        if not indices:
            raise FileNotFoundError(f"Could not load any indices for: {names}")
        logger.debug("Loaded indices for %d/%d books", len(indices), len(names))

        result = query_engine.ask_question_multi(
            indices,
            question,
            prompt_type=prompt_type,
            retrieval_k=retrieval_k,
            runtime_params=runtime_params or None,
        )
        meta = {
            "chunks": result["retrieved_chunks"],
            "prompt_type": prompt_type,
            "sources": [s["content_preview"] for s in result["sources"][:3]],
            "multi_book": True,
            "book_names": result.get("book_names", names),
        }
        if session_id:
            session_manager.add_exchange(session_id, question, result["response"], metadata=meta)
        else:
            label = "+".join(result.get("book_names", names))
            chat_history_manager.add_exchange(label, question, result["response"], metadata=meta)
        return {
            "success": True,
            "response": result["response"],
            "retrieved_chunks": result["retrieved_chunks"],
            "sources": result["sources"],
            "book_names": result.get("book_names", names),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Multi-book chat failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Chat: multi-book streaming SSE ───────────────────────────────────────────

@app.post("/api/chat/multi/stream")
async def stream_chat_multi(
    session_id: Optional[str] = Form(None),
    book_names: Optional[str] = Form(None),
    question: str = Form(...),
    prompt_type: str = Form("book_qa"),
    retrieval_k: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
):
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    if not book_names and session_id:
        session = session_manager.get_session(session_id)
        if session:
            sel = session.get("selected_books") or []
            book_names = ",".join(sel) if sel else None

    if not book_names:
        raise HTTPException(status_code=400, detail="book_names or session_id with selected_books is required")

    names = [n.strip() for n in book_names.split(",") if n.strip()]
    if not names:
        raise HTTPException(status_code=400, detail="At least one book name required")

    logger.info("POST /api/chat/multi/stream: books=%s, prompt_type=%s, query=%.60s", names, prompt_type, question)

    runtime_params: Dict[str, Any] = {}
    if temperature is not None:
        runtime_params["temperature"] = temperature
    if top_p is not None:
        runtime_params["top_p"] = top_p

    async def event_generator():
        full_response = ""
        sources = []
        try:
            indices = _load_book_indices(names)
            logger.debug("Multi-stream: loaded indices for %d/%d books", len(indices), len(names))
            if not indices:
                payload = json.dumps({"type": "error", "message": f"No indices found for: {names}", "code": 404})
                yield f"data: {payload}\n\n"
                return
        except Exception as exc:
            payload = json.dumps({"type": "error", "message": str(exc), "code": 500})
            yield f"data: {payload}\n\n"
            return

        event_count = 0
        for event in query_engine.stream_question_multi(
            indices,
            question,
            prompt_type=prompt_type,
            retrieval_k=retrieval_k,
            runtime_params=runtime_params or None,
        ):
            if event["type"] == "meta":
                sources = event.get("sources", [])
                logger.debug("Multi-stream: received %d sources", len(sources))
            elif event["type"] == "chunk":
                full_response += event.get("text", "")
            payload = json.dumps(event, ensure_ascii=False)
            yield f"data: {payload}\n\n"
            event_count += 1

        logger.info("Multi-stream complete: books=%s, events=%d, response_len=%d", names, event_count, len(full_response))

        if full_response:
            meta = {
                "chunks": len(sources),
                "prompt_type": prompt_type,
                "sources": [s["content_preview"] for s in sources[:3]],
                "multi_book": True,
                "book_names": names,
            }
            if session_id:
                session_manager.add_exchange(session_id, question, full_response, metadata=meta)
            else:
                label = "+".join(names)
                chat_history_manager.add_exchange(label, question, full_response, metadata=meta)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Chat history ───────────────────────────────────────────────────────────────

@app.get("/api/chat/history/{book_name}")
async def get_chat_history(book_name: str):
    logger.debug("GET /api/chat/history/%s", book_name)
    history = chat_history_manager.load_history(book_name)
    stats = chat_history_manager.get_stats(book_name)
    logger.debug("Chat history for %s: %d messages", book_name, len(history))
    return {"success": True, "book_name": book_name, "history": history, "stats": stats}


@app.delete("/api/chat/history/{book_name}")
async def clear_chat_history(book_name: str):
    logger.info("Clearing chat history for: %s", book_name)
    cleared = chat_history_manager.clear_history(book_name)
    logger.info("Chat history cleared for %s: %s", book_name, cleared)
    return {"success": True, "cleared": cleared, "book_name": book_name}


@app.get("/api/chat/history/{book_name}/export")
async def export_chat_history(book_name: str, format: str = "json"):
    if format not in ("json", "markdown"):
        raise HTTPException(status_code=400, detail="format must be 'json' or 'markdown'")
    content = chat_history_manager.export_history(book_name, fmt=format)
    media = "application/json" if format == "json" else "text/markdown"
    filename = f"{book_name}_chat_history.{format if format == 'json' else 'md'}"
    return StreamingResponse(
        iter([content]),
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)