"""
query_engine.py — Document retrieval + LLM answering with:
  • Single-book & multi-book retrieval
  • Parent-Child document resolution (children for vector search, parents for context)
  • Similarity or MMR (Max Marginal Relevance) search
  • Multi-index fusion via Reciprocal Rank Fusion (RRF) or score fusion
  • Cross-encoder re-ranking of parent documents
  • Optional hybrid search (BM25 + dense)
  • Rich source metadata (including source book name)
  • Streaming via model_provider.stream_chat()
"""

import logging
import os
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import hashlib
import config
from embedding_manager import BookEmbeddingManager
from model_provider import ModelProvider
from settings_manager import SettingsManager

logger = logging.getLogger(__name__)


class BookQueryEngine:
    def __init__(
        self,
        settings_manager: Optional[SettingsManager] = None,
        model_provider: Optional[ModelProvider] = None,
        embedding_manager: Optional[BookEmbeddingManager] = None,
    ) -> None:
        logger.info("Initializing BookQueryEngine...")
        self.settings_mgr = settings_manager or SettingsManager()
        self.model_provider = model_provider or ModelProvider(self.settings_mgr)
        self.embedding_mgr = embedding_manager or BookEmbeddingManager()
        logger.info("BookQueryEngine ready and optimized for Parent-Child retrieval.")

    # ═════════════════════════════════════════════════════════════════════════
    #  核心檢索與 Parent-Child 還原邏輯
    # ═════════════════════════════════════════════════════════════════════════

    def _resolve_parent_documents(self, child_docs_with_scores: List[Tuple[Document, float]], parent_store: Dict[str, dict]) -> List[Tuple[Document, float]]:
        """
        核心重構方法：將向量庫檢索出來的 Child 節點，完美還原為具備高階層上下文的 Parent 節點並去重
        """
        if not parent_store:
            logger.warning("Parent store empty or missing. Delivering raw child snippets instead.")
            return child_docs_with_scores

        seen_parent_ids = set()
        resolved_parents_with_scores = []
        raw_fallback_count = 0

        for child_doc, score in child_docs_with_scores:
            parent_id = child_doc.metadata.get("parent_id")
            
            if parent_id and parent_id in parent_store:
                if parent_id not in seen_parent_ids:
                    seen_parent_ids.add(parent_id)
                    parent_info = parent_store[parent_id]
                    
                    # 構造還原後的父文本 Document 物件
                    parent_doc = Document(
                        page_content=parent_info["text"],
                        metadata=parent_info["metadata"].copy()
                    )
                    # 保留子文本中特有的標示，便於前端追溯與 Reranker 識別
                    parent_doc.metadata["_child_retrieved_content"] = child_doc.page_content
                    if "source" not in parent_doc.metadata and "source" in child_doc.metadata:
                        parent_doc.metadata["source"] = child_doc.metadata["source"]
                    if "page" not in parent_doc.metadata and "page" in child_doc.metadata:
                        parent_doc.metadata["page"] = child_doc.metadata["page"]
                        
                    resolved_parents_with_scores.append((parent_doc, score))
            else:
                # 若無對應的父塊（例如舊有庫），保留子塊避免丟失資訊
                child_id = child_doc.metadata.get("child_id", "raw")
                if child_id not in seen_parent_ids:
                    seen_parent_ids.add(child_id)
                    resolved_parents_with_scores.append((child_doc, score))
                    raw_fallback_count += 1

        logger.debug(
            "Parent resolution: %d children in -> %d parents out (raw_fallback=%d, seen=%d)",
            len(child_docs_with_scores), len(resolved_parents_with_scores),
            raw_fallback_count, len(seen_parent_ids),
        )
        return resolved_parents_with_scores

    def _retrieve_single_index(
        self, db_id: str, query: str, top_k: int, search_type: str = "similarity"
    ) -> List[Tuple[Document, float]]:
        """在單個 FAISS 索引庫中查找 Child 塊，並將其升級還原為 Parent 塊"""
        logger.info("Retrieving from index %s (top_k=%d, search_type=%s)", db_id, top_k, search_type)
        try:
            faiss_db = self.embedding_mgr.load_faiss_index(db_id)
            parent_store = self.embedding_mgr.load_parent_store(db_id)
        except FileNotFoundError as exc:
            logger.error("Skipping missing index %s: %s", db_id, exc)
            return []

        # 由於子塊還原父塊會進行去重，為了保證最終數量，向量庫檢索時先放大 top_k 數量
        fetch_k = top_k * 3
        logger.debug("Fetch_k=%d (top_k=%d * 3) for dedup headroom", fetch_k, top_k)

        if search_type.lower() == "mmr":
            try:
                logger.debug("Using MMR search (k=%d, fetch_k=%d)", top_k, fetch_k)
                child_docs = faiss_db.max_marginal_relevance_search(query, k=top_k, fetch_k=fetch_k)
                child_with_scores = [(doc, 0.5) for doc in child_docs]
            except Exception as exc:
                logger.warning("MMR failed (%s), using standard similarity", exc)
                child_with_scores = faiss_db.similarity_search_with_score(query, k=fetch_k)
        else:
            logger.debug("Using similarity search (k=%d)", fetch_k)
            child_with_scores = faiss_db.similarity_search_with_score(query, k=fetch_k)

        logger.debug("Raw child results: %d", len(child_with_scores))

        # 將檢索出的 Child 映射還原至完整的 Parent 塊
        resolved_parents = self._resolve_parent_documents(child_with_scores, parent_store)
        
        # 截斷保留至最終用戶設定的 top_k 數量
        result = resolved_parents[:top_k]
        logger.info("Index %s: %d children -> %d parents (final top_k=%d)", db_id, len(child_with_scores), len(resolved_parents), len(result))
        return result

    # ═════════════════════════════════════════════════════════════════════════
    #  多庫融合與問答流程（維持原架構高水準接口不變）
    # ═════════════════════════════════════════════════════════════════════════

    def retrieve_merged_docs(
        self, db_ids: List[str], query: str, top_k: int, search_type: str = "similarity"
    ) -> List[Tuple[Document, float]]:
        """支援多個書籍索引庫的並行並發檢索與融合"""
        if not db_ids:
            logger.warning("retrieve_merged_docs called with empty db_ids")
            return []
        logger.info("Merging %d indices (top_k=%d, search_type=%s)", len(db_ids), top_k, search_type)
        if len(db_ids) == 1:
            result = self._retrieve_single_index(db_ids[0], query, top_k, search_type)
            logger.debug("Single index result: %d docs", len(result))
            return result

        all_results_dict = {}
        for db_id in db_ids:
            res = self._retrieve_single_index(db_id, query, top_k, search_type)
            all_results_dict[db_id] = res
            logger.debug("Index %s returned %d docs", db_id, len(res))

        fusion_method = self.settings_mgr.get("FUSION_METHOD", "score_fusion")
        logger.info("Fusing %d result sets using method: %s", len(all_results_dict), fusion_method)
        if fusion_method == "rrf":
            return self._reciprocal_rank_fusion(all_results_dict, top_k)
        return self._score_fusion_flat(all_results_dict, top_k)

    def query_book(
        self, book_name: str, query: str, prompt_key: str = "book_qa"
    ) -> Tuple[str, List[Dict]]:
        """單一書籍問答同步接口"""
        if not self.embedding_mgr:
            raise ValueError("EmbeddingManager instance is required.")
        book_id = self.embedding_mgr.clean_id(book_name)
        db_id = self.embedding_mgr.version_mgr.get_current_db_id(book_id)
        if not db_id:
            return f"No database index found for book '{book_name}'. Please upload first.", []

        return self.query_multi_books([db_id], query, prompt_key, override_book_name=book_name)

    def query_multi_books(
        self, db_ids: List[str], query: str, prompt_key: str = "multi_book_qa", override_book_name: str = ""
    ) -> Tuple[str, List[Dict]]:
        """多書籍問答同步接口"""
        top_k = self.settings_mgr.get("TOP_K", 4)
        search_type = self.settings_mgr.get("SEARCH_TYPE", "similarity")

        parents_with_scores = self.retrieve_merged_docs(db_ids, query, top_k, search_type)

        final_docs = [d for d, _ in parents_with_scores]
        final_scores = [s for _, s in parents_with_scores]

        context_str = "\n\n---\n\n".join([d.page_content for d in final_docs])
        sources = self._build_sources(final_docs, final_scores, override_book_name)

        answer = self.model_provider.generate_response(prompt_key, context=context_str, query=query)
        return answer, sources

    def stream_book(self, book_name: str, query: str, prompt_key: str = "book_qa") -> Iterator[Dict]:
        """單一書籍串流 SSE 核心生成接口"""
        if not self.embedding_mgr:
            yield {"event": "error", "data": "EmbeddingManager missing"}
            return
        book_id = self.embedding_mgr.clean_id(book_name)
        db_id = self.embedding_mgr.version_mgr.get_current_db_id(book_id)
        if not db_id:
            yield {"event": "error", "data": f"No index for '{book_name}'"}
            return

        yield from self.stream_multi_books([db_id], query, prompt_key, override_book_name=book_name)

    def stream_multi_books(
        self, db_ids: List[str], query: str, prompt_key: str = "multi_book_qa", override_book_name: str = ""
    ) -> Iterator[Dict]:
        """多書籍串流 SSE 核心生成接口"""
        top_k = self.settings_mgr.get("TOP_K", 4)
        search_type = self.settings_mgr.get("SEARCH_TYPE", "similarity")

        yield {"event": "status", "data": "Searching through book database index..."}
        parents_with_scores = self.retrieve_merged_docs(db_ids, query, top_k, search_type)

        final_docs = [d for d, _ in parents_with_scores]
        final_scores = [s for _, s in parents_with_scores]

        context_str = "\n\n---\n\n".join([d.page_content for d in final_docs])
        sources = self._build_sources(final_docs, final_scores, override_book_name)

        yield {"event": "sources", "data": sources}

        yield {"event": "status", "data": "Synthesizing answer with DeepSeek..."}
        for token in self.model_provider.stream_chat(query=query, context=context_str, feature=prompt_key):
            yield {"event": "token", "data": token}

        yield {"event": "done", "data": ""}

    # ═════════════════════════════════════════════════════════════════════════
    #  app.py 相容接口
    # ═════════════════════════════════════════════════════════════════════════

    def ask_question(
        self, book_name: str, query: str, vector_index=None,
        prompt_type: str = "book_qa", retrieval_k: int = None,
        runtime_params: dict = None, mode: str = "rag",
    ) -> Dict:
        logger.info("ask_question: book=%s, query=%.60s, prompt_type=%s, mode=%s", book_name, query, prompt_type, mode)
        if not self.embedding_mgr:
            raise ValueError("EmbeddingManager instance is required.")

        if mode == "full":
            full_content = self.embedding_mgr.get_full_file_content(book_name)
            sources = [{"index": 0, "content": full_content, "content_preview": full_content[:400], "source": book_name, "page": None, "score": 1.0, "book_name": book_name}]
            answer = self.model_provider.generate_response(prompt_type, context=full_content, query=query, runtime_params=runtime_params)
            logger.info("Full-file answer: %d chars", len(answer))
            return {"response": answer, "retrieved_chunks": 1, "sources": sources}

        db_id = self.embedding_mgr.resolve_db_id(book_name)

        top_k = retrieval_k or self.settings_mgr.get("TOP_K", 4)
        search_type = self.settings_mgr.get("SEARCH_TYPE", "similarity")
        logger.debug("top_k=%d, search_type=%s", top_k, search_type)

        parents_with_scores = self._retrieve_single_index(db_id, query, top_k, search_type)

        final_docs = [d for d, _ in parents_with_scores]
        final_scores = [s for _, s in parents_with_scores]

        context_str = "\n\n---\n\n".join([d.page_content for d in final_docs])
        sources = self._build_sources(final_docs, final_scores, book_name)
        logger.info("Built context: %d chars from %d docs", len(context_str), len(final_docs))

        answer = self.model_provider.generate_response(prompt_type, context=context_str, query=query, runtime_params=runtime_params)
        logger.info("Generated answer: %d chars", len(answer))

        return {
            "response": answer,
            "retrieved_chunks": len(final_docs),
            "sources": sources,
        }

    def stream_question(
        self, book_name: str, query: str, vector_index=None,
        prompt_type: str = "book_qa", retrieval_k: int = None,
        runtime_params: dict = None, mode: str = "rag",
    ) -> Iterator[Dict]:
        logger.info("stream_question: book=%s, query=%.60s, mode=%s", book_name, query, mode)
        if not self.embedding_mgr:
            logger.error("EmbeddingManager missing in stream_question")
            yield {"type": "error", "message": "EmbeddingManager missing"}
            return

        if mode == "full":
            try:
                full_content = self.embedding_mgr.get_full_file_content(book_name)
            except FileNotFoundError:
                yield {"type": "error", "message": f"Archived file not found for '{book_name}'"}
                return
            sources = [{"index": 0, "content": full_content, "content_preview": full_content[:400], "source": book_name, "page": None, "score": 1.0, "book_name": book_name}]
            yield {"type": "meta", "sources": sources}
            for token in self.model_provider.stream_chat(query=query, context=full_content, feature=prompt_type, runtime_params=runtime_params):
                yield {"type": "chunk", "text": token}
            yield {"type": "done", "text": ""}
            return

        try:
            db_id = self.embedding_mgr.resolve_db_id(book_name)
        except FileNotFoundError:
            logger.error("No index for book: %s", book_name)
            yield {"type": "error", "message": f"No index for '{book_name}'"}
            return

        top_k = retrieval_k or self.settings_mgr.get("TOP_K", 4)
        search_type = self.settings_mgr.get("SEARCH_TYPE", "similarity")
        logger.debug("top_k=%d, search_type=%s", top_k, search_type)

        parents_with_scores = self._retrieve_single_index(db_id, query, top_k, search_type)

        final_docs = [d for d, _ in parents_with_scores]
        final_scores = [s for _, s in parents_with_scores]

        context_str = "\n\n---\n\n".join([d.page_content for d in final_docs])
        sources = self._build_sources(final_docs, final_scores, book_name)
        logger.info("Stream context built: %d chars, %d sources", len(context_str), len(sources))

        yield {"type": "meta", "sources": sources}

        token_count = 0
        for token in self.model_provider.stream_chat(query=query, context=context_str, feature=prompt_type, runtime_params=runtime_params):
            yield {"type": "chunk", "text": token}
            token_count += 1

        logger.info("Stream done: %s yielded %d tokens", book_name, token_count)
        yield {"type": "done", "text": ""}

    def ask_question_multi(
        self, indices: List, query: str,
        prompt_type: str = "multi_book_qa", retrieval_k: int = None,
        runtime_params: dict = None,
    ) -> Dict:
        book_names = [name for name, _ in indices]
        logger.info("ask_question_multi: books=%s, query=%.60s", book_names, query)

        db_ids = []
        resolved_names = []
        for name, idx in indices:
            try:
                db_id = self.embedding_mgr.resolve_db_id(name)
                db_ids.append(db_id)
                resolved_names.append(name)
                logger.debug("Resolved %s -> db_id=%s", name, db_id)
            except FileNotFoundError:
                logger.warning("No db_id found for book: %s", name)

        if not db_ids:
            logger.error("No indices found for any of the provided books: %s", book_names)
            raise FileNotFoundError("No indices found for provided books")

        top_k = retrieval_k or self.settings_mgr.get("TOP_K", 4)
        search_type = self.settings_mgr.get("SEARCH_TYPE", "similarity")
        logger.debug("top_k=%d, search_type=%s, db_ids=%s", top_k, search_type, db_ids)

        parents_with_scores = self.retrieve_merged_docs(db_ids, query, top_k, search_type)

        final_docs = [d for d, _ in parents_with_scores]
        final_scores = [s for _, s in parents_with_scores]

        context_str = "\n\n---\n\n".join([d.page_content for d in final_docs])
        sources = self._build_sources(final_docs, final_scores, "+".join(resolved_names))
        logger.info("Multi-book context: %d chars from %d docs across %d books", len(context_str), len(final_docs), len(db_ids))

        answer = self.model_provider.generate_response(prompt_type, context=context_str, query=query, runtime_params=runtime_params)
        logger.info("Multi-book answer: %d chars", len(answer))

        return {
            "response": answer,
            "retrieved_chunks": len(final_docs),
            "sources": sources,
            "book_names": resolved_names,
        }

    def stream_question_multi(
        self, indices: List, query: str,
        prompt_type: str = "multi_book_qa", retrieval_k: int = None,
        runtime_params: dict = None,
    ) -> Iterator[Dict]:
        book_names = [name for name, _ in indices]
        logger.info("stream_question_multi: books=%s, query=%.60s", book_names, query)

        db_ids = []
        resolved_names = []
        for name, idx in indices:
            try:
                db_id = self.embedding_mgr.resolve_db_id(name)
                db_ids.append(db_id)
                resolved_names.append(name)
                logger.debug("Resolved %s -> db_id=%s", name, db_id)
            except FileNotFoundError:
                logger.warning("No db_id found for book: %s", name)

        if not db_ids:
            logger.error("No indices found for books: %s", book_names)
            yield {"type": "error", "message": "No indices found for provided books"}
            return

        top_k = retrieval_k or self.settings_mgr.get("TOP_K", 4)
        search_type = self.settings_mgr.get("SEARCH_TYPE", "similarity")
        logger.debug("top_k=%d, search_type=%s", top_k, search_type)

        parents_with_scores = self.retrieve_merged_docs(db_ids, query, top_k, search_type)

        final_docs = [d for d, _ in parents_with_scores]
        final_scores = [s for _, s in parents_with_scores]

        context_str = "\n\n---\n\n".join([d.page_content for d in final_docs])
        sources = self._build_sources(final_docs, final_scores, "+".join(resolved_names))
        logger.info("Multi-stream context: %d chars, %d sources", len(context_str), len(sources))

        yield {"type": "meta", "sources": sources}

        token_count = 0
        for token in self.model_provider.stream_chat(query=query, context=context_str, feature=prompt_type, runtime_params=runtime_params):
            yield {"type": "chunk", "text": token}
            token_count += 1

        logger.info("Multi-stream done: %s yielded %d tokens", resolved_names, token_count)
        yield {"type": "done", "text": ""}

    def _score_fusion_flat(self, all_results: Dict[str, List[Tuple[Document, float]]], top_k: int) -> List[Tuple[Document, float]]:
        flat_list = []
        for db_id, items in all_results.items():
            for doc, score in items:
                doc.metadata["_db_id"] = db_id
                flat_list.append((doc, score))
        # 依分數由小到大排序（FAISS 預設 L2 距離越小越優）
        flat_list.sort(key=lambda x: x[1])
        result = flat_list[:top_k]
        logger.debug("Score fusion: %d raw -> %d fused (top_k=%d)", len(flat_list), len(result), top_k)
        return result

    def _reciprocal_rank_fusion(self, all_results: Dict[str, List[Tuple[Document, float]]], top_k: int, c: int = 60) -> List[Tuple[Document, float]]:
        rrf_scores = defaultdict(float)
        doc_map = {}
        
        total_input = 0
        for db_id, items in all_results.items():
            total_input += len(items)
            for rank, (doc, _) in enumerate(items):
                # 建立能跨庫去重的唯一文本 Key
                content_hash = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest() if hasattr(hashlib, 'md5') else str(hash(doc.page_content))
                if content_hash not in doc_map:
                    doc.metadata["_db_id"] = db_id
                    doc_map[content_hash] = doc
                rrf_scores[content_hash] += 1.0 / (c + rank + 1)

        sorted_hashes = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        result = [(doc_map[h], float(rrf_scores[h])) for h in sorted_hashes[:top_k]]
        logger.debug("RRF fusion: %d items from %d indexes -> %d fused (unique=%d, top_k=%d)",
                     total_input, len(all_results), len(result), len(doc_map), top_k)
        return result

    def _build_sources(self, docs: List, scores: List, book_name: str = "") -> List[Dict]:
        result = []
        for i, doc in enumerate(docs):
            score = scores[i] if i < len(scores) else None
            result.append(self._doc_to_source(doc, score, book_name))
        return result

    @staticmethod
    def _doc_to_source(doc, score, book_name: str = "") -> Dict:
        source_book = book_name or doc.metadata.get("_book_name", "") or doc.metadata.get("book_name", "Spec_Book")
        return {
            "index": 0,
            "content": doc.page_content,
            "content_preview": doc.page_content[:400],
            "source": doc.metadata.get("source", ""),
            "page": doc.metadata.get("page", None),
            "score": round(float(score), 4) if score is not None else 0.0,
            "book_name": source_book
        }