import glob
import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from functools import wraps
from typing import Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config
import document_loader
from version_manager import BookVersionManager

logger = logging.getLogger(__name__)


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}...")
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {elapsed:.2f}s: {e}")
            raise
    return wrapper

class BookEmbeddingManager:
    def __init__(self):
        logger.info("=" * 60)
        logger.info("Initializing Optimized Hardware-RAG Embedding Manager...")
        try:
            start_time = time.time()
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
            logger.info(f"Device: {config.EMBEDDING_DEVICE}")

            # 針對 CPU 進行底層多執行緒並行優化，防止 BGE-M3 執行緒過度切換
            if config.EMBEDDING_DEVICE.lower() == "cpu":
                import torch
                torch.set_num_threads(4)
                logger.info("PyTorch CPU threads set to 4 for optimal matrix computing.")
            else:
                logger.warning("Non-CPU device detected. Ensure that the embedding model is compatible and properly configured.")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={"device": config.EMBEDDING_DEVICE},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 32  # CPU 批次優化
                },
            )

            elapsed = time.time() - start_time
            logger.info(f"EmbeddingManager initialized successfully in {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingManager: {e}")
            raise

        self.version_mgr = BookVersionManager()
        self.index_base_dir = config.VECTOR_DB_DIR
        os.makedirs(self.index_base_dir, exist_ok=True)

    @staticmethod
    def clean_id(name: str) -> str:
        """將書籍名稱清洗為合法的文件夾/標識符名稱"""
        cleaned = re.sub(r"[^\w\s-]", "", name)
        return re.sub(r"[\s-]+", "_", cleaned).strip("_")

    def _create_parent_child_chunks(self, documents: List[Document]) -> Tuple[List[Document], Dict[str, dict]]:
        """
        針對晶片技術規格書設計的雙層切片器
        將文檔切分為用於閱讀推理的 Parent 塊與用於高速向量檢索的 Child 塊
        """
        # 父文本塊：保留完整的暫存器定義、狀態機說明或代碼上下文
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "]
        )
        
        # 子文本塊：高精度的關鍵字與語意特徵捕捉（短小精悍，大幅加速 CPU 向量運算）
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n", " ", ""]
        )

        logger.info("Splitting raw documents into Parent chunks...")
        parent_docs = parent_splitter.split_documents(documents)

        # Fallback: if parent-child split yields nothing, use normal single-level chunking
        if not parent_docs:
            logger.warning("Parent split produced 0 chunks — falling back to normal RAG chunking")
            normal_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
            )
            normal_docs = normal_splitter.split_documents(documents)
            child_docs = []
            parent_store = {}
            for i, doc in enumerate(normal_docs):
                pid = f"p_{uuid.uuid4().hex[:12]}"
                parent_store[pid] = {"text": doc.page_content, "metadata": doc.metadata}
                c_meta = doc.metadata.copy()
                c_meta["parent_id"] = pid
                c_meta["child_id"] = f"c_{pid}_0"
                child_docs.append(Document(page_content=doc.page_content, metadata=c_meta))
            logger.info("Fallback normal RAG: %d chunks", len(child_docs))
            return child_docs, parent_store

        child_documents = []
        parent_store = {}

        logger.info(f"Generated {len(parent_docs)} Parent chunks. Generating Child chunks...")
        for p_doc in parent_docs:
            parent_id = f"p_{uuid.uuid4().hex[:12]}"
            
            # 提取元數據
            page_info = p_doc.metadata.get("page", 0)
            source_info = os.path.basename(p_doc.metadata.get("source", "Unknown_Spec"))
            
            # 將父文本完整的資訊存入對照字典中
            parent_store[parent_id] = {
                "text": p_doc.page_content,
                "metadata": p_doc.metadata
            }
            
            # 對當前的父文本塊進行二次細切
            sub_chunks = child_splitter.split_text(p_doc.page_content)
            
            for c_idx, child_text in enumerate(sub_chunks):
                child_id = f"c_{parent_id}_{c_idx}"
                
                # 複製父元數據，並注入關鍵的雙向映射標記
                child_metadata = p_doc.metadata.copy()
                child_metadata["parent_id"] = parent_id
                child_metadata["child_id"] = child_id
                
                # 注入上下文前綴，確保 Child 在向量空間中帶有明確的硬體手冊出處脈絡
                context_prefix = f"[{source_info} P{page_info}]: "
                full_child_content = context_prefix + child_text
                
                c_doc = Document(
                    page_content=full_child_content,
                    metadata=child_metadata
                )
                child_documents.append(c_doc)
                
        return child_documents, parent_store

    @timed
    def build_book_index(self, file_path: str, book_name: str, db_id: str = None) -> str:
        """
        為上傳的晶片設計書籍或規格書建立 Parent-Child 雙層向量索引架構
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading document into reader: {file_path}")
        raw_documents = document_loader.load_document(file_path)
        
        # 實施專有的雙層切片設計
        child_docs, parent_store = self._create_parent_child_chunks(raw_documents)
        logger.info(f"Final Count -> Parents: {len(parent_store)} | Childs: {len(child_docs)}")

        # 分配唯一的數據庫庫 ID (always ASCII-safe)
        db_id = f"db_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        idx_dir = os.path.join(self.index_base_dir, db_id)
        os.makedirs(idx_dir, exist_ok=True)

        # 1. 僅對短小精悍的 Child Chunks 進行向量化建庫 (大幅縮短 CPU 耗時)
        logger.info("Embedding Child chunks into FAISS vector space...")
        faiss_db = FAISS.from_documents(child_docs, self.embeddings)
        faiss_db.save_local(idx_dir)

        # 2. 將 Parent 還原對照表儲存為本地 JSON
        parent_store_path = os.path.join(idx_dir, "parent_store.json")
        with open(parent_store_path, "w", encoding="utf-8") as f:
            json.dump(parent_store, f, ensure_ascii=False, indent=2)

        # 3. 建立並寫入索引元數據
        meta_file = os.path.join(idx_dir, "book_meta.json")
        meta_data = {
            "book_name": book_name,
            "db_id": db_id,
            "file_type": os.path.splitext(file_path)[-1].lower(),
            "parent_chunks_count": len(parent_store),
            "child_chunks_count": len(child_docs),
            "created_at": time.time(),
            "file_size_bytes": os.path.getsize(file_path)
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Successfully generated database index index id: {db_id}")
        return db_id

    def load_faiss_index(self, db_id: str) -> FAISS:
        """加載指定數據庫 ID 的 FAISS 向量數據庫"""
        idx_dir = os.path.join(self.index_base_dir, db_id)
        if not os.path.exists(idx_dir):
            logger.error("FAISS index folder missing: %s", idx_dir)
            raise FileNotFoundError(f"FAISS index folder missing: {idx_dir}")
        logger.info("Loading FAISS index: %s", idx_dir)
        index = FAISS.load_local(idx_dir, self.embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded: %s (ntotal=%d)", db_id, index.index.ntotal)
        return index

    def load_parent_store(self, db_id: str) -> Dict[str, dict]:
        """加載指定數據庫 ID 的 Parent 文本映射對照表"""
        store_path = os.path.join(self.index_base_dir, db_id, "parent_store.json")
        if not os.path.exists(store_path):
            logger.warning("parent_store.json missing in index %s. Falling back to child-only.", db_id)
            return {}
        with open(store_path, "r", encoding="utf-8") as f:
            parent_store = json.load(f)
        logger.info("Loaded parent_store for %s: %d parent entries", db_id, len(parent_store))
        return parent_store

    def remove_book_index(self, book_name: str) -> bool:
        """刪除某一本書名對應的所有本地向量與對照數據"""
        logger.info("Removing book index for: %s", book_name)
        book_id = self.clean_id(book_name)
        history = self.version_mgr.get_version_history(book_id)
        metadata_book_id = book_id

        # Fallback: scan metadata files by book name when id lookup returns empty
        if not history:
            logger.warning("No history found via id='%s', scanning metadata by book name", book_id)
            for meta_file in glob.glob(os.path.join(self.version_mgr.metadata_dir, "*_book_meta.json")):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("book_name") == book_name:
                        history = meta.get("versions", [])
                        raw_id = os.path.basename(meta_file)
                        metadata_book_id = raw_id[:raw_id.index("_book_meta.json")]
                        logger.info("Found metadata via scan: %s (%d versions, id=%s)", meta_file, len(history), metadata_book_id)
                        break
                except (json.JSONDecodeError, IOError):
                    continue

        removed = False
        for version in history:
            db_id = version.get("db_id")
            if not db_id:
                logger.warning("Version entry missing db_id, skipping")
                continue
            idx_dir = os.path.join(self.index_base_dir, db_id)
            if os.path.exists(idx_dir):
                try:
                    shutil.rmtree(idx_dir)
                    logger.info("Purged index directory: %s", idx_dir)
                    removed = True
                except Exception as e:
                    logger.error("Error deleting index directory %s: %s", idx_dir, e)

        self.version_mgr.delete_book_metadata(metadata_book_id)
        logger.info("Book '%s' removal complete (versions=%d, removed=%s)", book_name, len(history), removed)
        return removed

    def get_book_info(self, book_name: str) -> dict:
        """獲取書籍的最新元數據資訊"""
        logger.info("Getting book info for: %s", book_name)
        book_id = self.clean_id(book_name)
        db_id = self.version_mgr.get_current_db_id(book_id)
        logger.debug("Book ID: %s, DB ID: %s", book_id, db_id)

        if db_id:
            idx_dir = os.path.join(self.index_base_dir, db_id)
            meta_file = os.path.join(idx_dir, "book_meta.json")
            if os.path.exists(meta_file):
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                logger.info("Found book info via db_id=%s: %s", db_id, meta.get("book_name"))
                return meta
            logger.debug("meta_file not found at %s, falling back to scan", meta_file)

        for meta_file in glob.glob(os.path.join(self.index_base_dir, "*", "book_meta.json")):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("book_name") == book_name:
                    logger.info("Found book info via scan: %s", meta_file)
                    return meta
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Skipping unreadable meta file %s: %s", meta_file, e)
                continue

        logger.error("No book info found for: %s", book_name)
        raise FileNotFoundError(f"No information found for book '{book_name}'")

    def list_books(self) -> List[Dict]:
        books = []
        scan_path = os.path.join(self.index_base_dir, "*", "book_meta.json")
        logger.debug("Scanning for book metadata: %s", scan_path)
        for meta_file in glob.glob(scan_path):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                file_size = meta.get("file_size_bytes", 0)
                books.append({
                    "name": meta.get("book_name", "Unknown"),
                    "db_id": meta.get("db_id", ""),
                    "file_type": meta.get("file_type", ".txt"),
                    "chunk_count": meta.get("child_chunks_count", 0),
                    "created_at": meta.get("created_at", 0),
                    "size_kb": round(file_size / 1024, 1) if file_size else 0,
                })
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Skipping unreadable meta file %s: %s", meta_file, e)
                continue
        logger.info("Listed %d books from %s", len(books), self.index_base_dir)
        return books

    def ingest_book(self, file_path: str, book_name: str, source_filename: str = None, chunk_size: int = None, chunk_overlap: int = None) -> Dict:
        logger.info("Ingesting book: book_name=%s, source=%s, file=%s", book_name, source_filename, file_path)
        if chunk_size:
            logger.debug("Overriding chunk_size=%d, chunk_overlap=%d", chunk_size, chunk_overlap)

        book_id = self.clean_id(book_name)
        check = self.version_mgr.check_version(file_path, book_id, base_dir=self.index_base_dir)
        logger.info("Version check result: action=%s, version=%s, is_new=%s",
                     check.get("action"), check.get("version_number"), check.get("is_new"))

        file_type = os.path.splitext(file_path)[-1].lower()
        if not check["is_new"]:
            parent_store = self.load_parent_store(check["db_id"])
            parent_count = len(parent_store) if parent_store else 0
            logger.info("Reusing existing index: db_id=%s, parents=%d", check["db_id"], parent_count)
            return {
                "book_name": book_name,
                "action": check["action"],
                "version_number": check["version_number"],
                "message": check["message"],
                "chunk_count": parent_count,
                "file_type": file_type,
            }

        db_id = self.build_book_index(file_path, book_name)
        logger.info("Index built: db_id=%s", db_id)

        skip_commit = check.get("_internal", {}).get("skip_commit", False)
        if skip_commit:
            metadata = self.version_mgr.load_metadata(book_id)
            for v in metadata["versions"]:
                if v["hash"] == check["file_hash"]:
                    v["db_id"] = db_id
                    break
            metadata["current_db_id"] = db_id
            self.version_mgr.save_metadata(book_id, metadata)
            logger.info("Repaired: updated metadata db_id to %s for book=%s", db_id, book_id)
        else:
            check["db_id"] = db_id
            self.version_mgr.commit_new_version(book_id, check)
            logger.info("Version committed: book_id=%s, version=%s, db_id=%s", book_id, check["version_number"], db_id)

        parent_store = self.load_parent_store(db_id)
        logger.info("Ingest complete: book=%s, action=%s, parents=%d", book_name, check["action"], len(parent_store))
        return {
            "book_name": book_name,
            "action": check["action"],
            "version_number": check["version_number"],
            "message": check["message"],
            "chunk_count": len(parent_store),
            "file_type": file_type,
        }

    def delete_book(self, book_name: str) -> bool:
        logger.info("Deleting book: %s", book_name)
        result = self.remove_book_index(book_name)
        logger.info("Delete result for '%s': %s", book_name, result)
        return result

    def get_full_file_content(self, book_name: str) -> str:
        """Load the full archived file content for a book."""
        logger.info("Loading full file content for: %s", book_name)
        book_id = self.clean_id(book_name)
        metadata = self.version_mgr.load_metadata(book_id)
        current_hash = metadata.get("current_hash")
        versions = metadata.get("versions", [])

        archive_path = None
        if current_hash:
            for v in reversed(versions):
                if v.get("hash") == current_hash:
                    archive_path = v.get("archive_path")
                    break
        if not archive_path and versions:
            archive_path = versions[-1].get("archive_path")

        if not archive_path or not os.path.exists(archive_path):
            logger.error("Archived file not found for: %s", book_name)
            raise FileNotFoundError(f"Archived file not found for '{book_name}'")

        logger.info("Loading archived file: %s", archive_path)
        ext = os.path.splitext(archive_path)[1].lower()
        if document_loader.is_text_like(archive_path):
            with open(archive_path, "r", encoding="utf-8") as f:
                full_text = f.read()
        else:
            docs = document_loader.load_document(archive_path)
            full_text = "\n".join(d.page_content for d in docs)

        logger.info("Full file content loaded: %d chars for '%s'", len(full_text), book_name)
        return full_text

    def get_vector_index(self, book_name: str) -> FAISS:
        logger.info("Getting vector index for book: %s", book_name)
        book_id = self.clean_id(book_name)
        db_id = self.version_mgr.get_current_db_id(book_id)

        if db_id:
            idx_dir = os.path.join(self.index_base_dir, db_id)
            if os.path.exists(idx_dir):
                logger.debug("Resolved db_id=%s for book=%s (via metadata)", db_id, book_name)
                return self.load_faiss_index(db_id)
            logger.warning("Metadata db_id=%s directory missing, scanning index directories", db_id)

        for meta_file in glob.glob(os.path.join(self.index_base_dir, "*", "book_meta.json")):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("book_name") == book_name:
                    actual_db_id = meta.get("db_id")
                    if not actual_db_id:
                        continue
                    actual_dir = os.path.join(self.index_base_dir, actual_db_id)
                    if os.path.exists(actual_dir):
                        logger.info("Found index for '%s' via scan: db_id=%s", book_name, actual_db_id)
                        return self.load_faiss_index(actual_db_id)
            except (json.JSONDecodeError, IOError):
                continue

        logger.error("No index found for book: %s", book_name)
        raise FileNotFoundError(f"No index found for book '{book_name}'")

    def resolve_db_id(self, book_name: str) -> str:
        """Resolve db_id for a book, with scan fallback when version metadata is stale."""
        logger.debug("resolve_db_id: book=%s", book_name)
        book_id = self.clean_id(book_name)
        db_id = self.version_mgr.get_current_db_id(book_id)

        if db_id:
            idx_dir = os.path.join(self.index_base_dir, db_id)
            if os.path.exists(idx_dir):
                logger.debug("Resolved db_id=%s (via metadata)", db_id)
                return db_id
            logger.warning("Metadata db_id=%s missing on disk, scanning indexes", db_id)

        for meta_file in glob.glob(os.path.join(self.index_base_dir, "*", "book_meta.json")):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("book_name") == book_name:
                    actual = meta.get("db_id")
                    if actual and os.path.exists(os.path.join(self.index_base_dir, actual)):
                        logger.info("Resolved db_id=%s (via scan)", actual)
                        return actual
            except (json.JSONDecodeError, IOError):
                continue

        logger.error("Cannot resolve db_id for book: %s", book_name)
        raise FileNotFoundError(f"No index found for book '{book_name}'")