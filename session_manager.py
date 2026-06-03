"""
session_manager.py — Independent chat sessions.

Sessions decouple chat history from books:
  - A session has a name, selected book(s), and message history.
  - Books can be selected/changed within a session without losing history.
  - Multiple sessions can coexist, each with different books and history.
"""

import json
import logging
import os
import time
import uuid
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

SESSION_DIR = config.SESSION_DIR
MAX_MESSAGES_PER_SESSION = 500


class SessionManager:
    """CRUD for chat sessions. Each session is a dir with meta.json + messages.json."""

    # ── Session CRUD ──────────────────────────────────────────────────────────

    def create_session(self, name: str, selected_books: Optional[List[str]] = None) -> Dict:
        session_id = uuid.uuid4().hex[:12]
        now = time.time()
        meta = {
            "id": session_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "selected_books": selected_books or [],
            "active_book": (selected_books or [None])[0],
        }
        self._save_meta(meta)
        self._save_messages(session_id, [])
        logger.info("Created session '%s' (id=%s)", name, session_id)
        return self._public_meta(meta)

    def list_sessions(self) -> List[Dict]:
        sessions = []
        if not os.path.isdir(SESSION_DIR):
            return sessions
        for entry in sorted(os.listdir(SESSION_DIR), reverse=True):
            meta_path = os.path.join(SESSION_DIR, entry, "meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    msg_path = os.path.join(SESSION_DIR, entry, "messages.json")
                    msg_count = 0
                    if os.path.exists(msg_path):
                        try:
                            with open(msg_path, "r", encoding="utf-8") as f:
                                msg_count = len(json.load(f))
                        except Exception:
                            pass
                    m = self._public_meta(meta)
                    m["message_count"] = msg_count
                    sessions.append(m)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Skipping corrupt session %s: %s", entry, exc)
        sessions.sort(key=lambda s: s.get("updated_at", 0), reverse=True)
        return sessions

    def get_session(self, session_id: str) -> Optional[Dict]:
        meta = self._load_meta(session_id)
        if meta is None:
            return None
        return self._public_meta(meta)

    def update_session(self, session_id: str, updates: Dict) -> Optional[Dict]:
        meta = self._load_meta(session_id)
        if meta is None:
            return None
        for key in ("name", "selected_books", "active_book"):
            if key in updates:
                meta[key] = updates[key]
        meta["updated_at"] = time.time()
        self._save_meta(meta)
        return self._public_meta(meta)

    def delete_session(self, session_id: str) -> bool:
        path = self._session_path(session_id)
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
            logger.info("Deleted session %s", session_id)
            return True
        return False

    # ── Messages ──────────────────────────────────────────────────────────────

    def load_messages(self, session_id: str) -> List[Dict]:
        return self._load_messages(session_id)

    def add_message(self, session_id: str, msg: Dict) -> Dict:
        messages = self._load_messages(session_id)
        ts = time.time()
        if "id" not in msg:
            msg["id"] = f"{int(ts * 1000)}_{len(messages)}"
        if "timestamp" not in msg:
            msg["timestamp"] = ts
        messages.append(msg)
        if len(messages) > MAX_MESSAGES_PER_SESSION:
            messages = messages[-MAX_MESSAGES_PER_SESSION:]
        self._save_messages(session_id, messages)

        meta = self._load_meta(session_id)
        if meta:
            meta["updated_at"] = time.time()
            self._save_meta(meta)
        return msg

    def add_exchange(
        self, session_id: str, question: str, answer: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        ts = time.time()
        base_id = f"{int(ts * 1000)}_{uuid.uuid4().hex[:6]}"
        user_msg = {
            "id": f"{base_id}_u", "role": "user",
            "content": question, "timestamp": ts,
        }
        asst_msg = {
            "id": f"{base_id}_a", "role": "assistant",
            "content": answer, "timestamp": ts,
            "metadata": metadata or {},
        }
        messages = self._load_messages(session_id)
        messages.extend([user_msg, asst_msg])
        if len(messages) > MAX_MESSAGES_PER_SESSION:
            messages = messages[-MAX_MESSAGES_PER_SESSION:]
        self._save_messages(session_id, messages)

        meta = self._load_meta(session_id)
        if meta:
            meta["updated_at"] = time.time()
            self._save_meta(meta)
        return asst_msg

    def clear_messages(self, session_id: str) -> bool:
        self._save_messages(session_id, [])
        return True

    def export_messages(self, session_id: str, fmt: str = "json") -> str:
        messages = self._load_messages(session_id)
        meta = self._load_meta(session_id) or {}
        session_name = meta.get("name", session_id)
        if fmt == "markdown":
            lines = [f"# Chat History — {session_name}", ""]
            for msg in messages:
                role_label = "**You**" if msg["role"] == "user" else "**AI Assistant**"
                ts_str = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(msg.get("timestamp", 0))
                )
                lines.append(f"{role_label} _{ts_str}_")
                lines.append("")
                lines.append(msg["content"])
                lines.append("")
                lines.append("---")
                lines.append("")
            return "\n".join(lines)
        return json.dumps(messages, ensure_ascii=False, indent=2)

    def get_stats(self, session_id: str) -> Dict:
        messages = self._load_messages(session_id)
        user_msgs = [m for m in messages if m["role"] == "user"]
        return {
            "total_messages": len(messages),
            "exchanges": len(user_msgs),
            "first_at": messages[0]["timestamp"] if messages else None,
            "last_at": messages[-1]["timestamp"] if messages else None,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _session_path(session_id: str) -> str:
        return os.path.join(SESSION_DIR, session_id)

    def _meta_path(self, session_id: str) -> str:
        return os.path.join(self._session_path(session_id), "meta.json")

    def _messages_path(self, session_id: str) -> str:
        return os.path.join(self._session_path(session_id), "messages.json")

    def _save_meta(self, meta: Dict) -> None:
        path = self._meta_path(meta["id"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _load_meta(self, session_id: str) -> Optional[Dict]:
        path = self._meta_path(session_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _save_messages(self, session_id: str, messages: List[Dict]) -> None:
        path = self._messages_path(session_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

    def _load_messages(self, session_id: str) -> List[Dict]:
        path = self._messages_path(session_id)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    @staticmethod
    def _public_meta(meta: Dict) -> Dict:
        return {
            "id": meta.get("id", ""),
            "name": meta.get("name", "Unnamed"),
            "created_at": meta.get("created_at", 0),
            "updated_at": meta.get("updated_at", 0),
            "selected_books": meta.get("selected_books", []),
            "active_book": meta.get("active_book"),
        }
