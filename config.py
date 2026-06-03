"""
config.py — Central configuration.  All tuneable values come from .env first,
then fall back to safe defaults.  Do NOT import project modules here.
"""

import logging
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, f"rag_{datetime.now().strftime('%Y%m%d')}.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── Embedding model ────────────────────────────────────────────────────────────
EMBEDDING_MODEL_FALLBACK = os.getenv("EMBEDDING_MODEL_FALLBACK")
EMBEDDING_MODEL= os.getenv("EMBEDDING_MODEL")

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "1200"))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "200"))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "250"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── LLM / API ─────────────────────────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com/v1")
MODEL_NORMAL = os.getenv("MODEL_NAME", "deepseek-v4-flash") 
MODEL_REASONING = os.getenv("MODEL_REASONING", "deepseek-v4-pro")
AI_PROVIDER = os.getenv("AI_PROVIDER", "deepseek")
TIMEOUT = int(os.getenv("TIMEOUT", "120"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_dbs")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ARCHIVE_DIR = os.path.join(BASE_DIR, "archives")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
CHAT_HISTORY_DIR = os.path.join(BASE_DIR, "chat_history")
SESSION_DIR = os.path.join(CHAT_HISTORY_DIR, "sessions")

# ── RAG ───────────────────────────────────────────────────────────────────────
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "20"))
MAX_VERSIONS_PER_FILE = int(os.getenv("MAX_VERSIONS_PER_FILE", "10"))
MAX_BOOK_BATCH_SIZE = int(os.getenv("MAX_BOOK_BATCH_SIZE", "25"))
MAX_BOOKS_PER_QUERY = int(os.getenv("MAX_BOOKS_PER_QUERY", "5"))        # max books to query simultaneously
RETRIEVAL_K_PER_BOOK = int(os.getenv("RETRIEVAL_K_PER_BOOK", "10"))    # k per book in multi-book mode
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() == "true"
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))                # weight for dense (0=pure sparse, 1=pure dense)

# ── Supported file types (delegated to document_loader) ───────────────────────
# Actual loader resolution happens in document_loader.py; this list is used
# for quick front-end validation and UI display.
SUPPORTED_EXTENSIONS = [
    ".txt", ".md", ".rst", ".text", ".log",
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".css", ".scss", ".less",
    ".sql", ".sh", ".bat", ".ps1",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".svg", ".tex",
    ".pdf",
    ".csv", ".tsv",
    ".json",
    ".html", ".htm", ".xhtml",
    ".docx",
    ".pptx",
    ".epub",
    ".rtf",
]
# Comma-separated string for HTML accept attribute
ACCEPT_EXTENSIONS_STR = ",".join(SUPPORTED_EXTENSIONS)

# ── Misc ───────────────────────────────────────────────────────────────────────
HASH_ALGORITHM = "sha256"
VERSION_TRACKING_ENABLED = True

# ── Bootstrap directories ──────────────────────────────────────────────────────
for _dir in [VECTOR_DB_DIR, UPLOAD_DIR, ARCHIVE_DIR, METADATA_DIR,
             CHAT_HISTORY_DIR, SESSION_DIR, LOG_DIR]:
    os.makedirs(_dir, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    for noisy in ("httpx", "openai", "urllib3", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


setup_logging()