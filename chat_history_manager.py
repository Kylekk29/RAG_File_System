"""
chat_history_manager.py — Per-book persistent chat history with export support.
Histories are stored as JSON files under BASE_DIR/chat_history/.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

CHAT_HISTORY_DIR = os.path.join(config.BASE_DIR, "chat_history")
MAX_MESSAGES_PER_BOOK = 300   # hard cap before oldest messages are pruned
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)


def _safe_name(book_name: str) -> str:
    return (
        book_name
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


class ChatHistoryManager:
    """Manages read/write of chat histories, one JSON file per book."""

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_history(self, book_name: str) -> List[Dict]:
        path = self._path(book_name)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.error("load_history(%s) failed: %s", book_name, exc)
            return []

    def save_history(self, book_name: str, history: List[Dict]) -> None:
        if len(history) > MAX_MESSAGES_PER_BOOK:
            history = history[-MAX_MESSAGES_PER_BOOK:]
        path = self._path(book_name)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(history, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error("save_history(%s) failed: %s", book_name, exc)

    def add_exchange(
        self,
        book_name: str,
        question: str,
        answer: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Append a user/assistant pair atomically and return the assistant entry."""
        history = self.load_history(book_name)
        ts = time.time()
        base_id = f"{int(ts * 1000)}_{len(history)}"

        user_msg: Dict = {
            "id": f"{base_id}_u",
            "role": "user",
            "content": question,
            "timestamp": ts,
        }
        asst_msg: Dict = {
            "id": f"{base_id}_a",
            "role": "assistant",
            "content": answer,
            "timestamp": ts,
            "metadata": metadata or {},
        }
        history.extend([user_msg, asst_msg])
        self.save_history(book_name, history)
        logger.debug("Saved exchange for '%s' (%d msgs total)", book_name, len(history))
        return asst_msg

    def clear_history(self, book_name: str) -> bool:
        path = self._path(book_name)
        if os.path.exists(path):
            os.remove(path)
            logger.info("Cleared history for '%s'", book_name)
            return True
        return False

    def list_books_with_history(self) -> List[str]:
        result = []
        for fname in os.listdir(CHAT_HISTORY_DIR):
            if fname.endswith("_history.json"):
                result.append(fname[: -len("_history.json")])
        return result

    def export_history(self, book_name: str, fmt: str = "json") -> str:
        history = self.load_history(book_name)
        if fmt == "markdown":
            lines = [f"# Chat History — {book_name}", ""]
            for msg in history:
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
        # default: JSON
        return json.dumps(history, ensure_ascii=False, indent=2)

    def get_stats(self, book_name: str) -> Dict:
        history = self.load_history(book_name)
        user_msgs = [m for m in history if m["role"] == "user"]
        return {
            "total_messages": len(history),
            "exchanges": len(user_msgs),
            "first_at": history[0]["timestamp"] if history else None,
            "last_at": history[-1]["timestamp"] if history else None,
        }

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _path(book_name: str) -> str:
        return os.path.join(CHAT_HISTORY_DIR, f"{_safe_name(book_name)}_history.json")