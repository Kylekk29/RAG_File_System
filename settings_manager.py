"""
settings_manager.py — Persistent runtime settings for model & RAG parameters.
All settings survive server restarts and are exposed via /api/settings endpoints.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import config

logger = logging.getLogger(__name__)

SETTINGS_FILE = os.path.join(config.BASE_DIR, "rag_settings.json")

# ── Provider presets ───────────────────────────────────────────────────────────
PROVIDER_PRESETS: Dict[str, Dict] = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash", "deepseek-v4-pro"],
        "default_model": "deepseek-chat",
        "supports_top_p": True,
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini",
        "supports_top_p": True,
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "default_model": "llama-3.3-70b-versatile",
        "supports_top_p": True,
    },
    "ollama": {
        "label": "Ollama (local)",
        "base_url": "http://localhost:11434/v1",
        "models": ["llama3.2", "mistral", "qwen2.5", "phi4", "deepseek-r1"],
        "default_model": "llama3.2",
        "supports_top_p": True,
    },
    "custom": {
        "label": "Custom / Other",
        "base_url": "",
        "models": [],
        "default_model": "",
        "supports_top_p": True,
    },
}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "model": {
        "provider": config.AI_PROVIDER,
        "model_name": "deepseek-chat",
        "base_url": config.BASE_URL,
        # api_key intentionally NOT stored here; read from env at runtime
        "temperature": 0.5,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "timeout": config.TIMEOUT,
        "max_retries": config.MAX_RETRIES,
    },
    "rag": {
        "retrieval_k": config.RETRIEVAL_K,
        "search_type": "similarity",   # "similarity" | "mmr"
        "mmr_lambda": 0.5,             # diversity factor for MMR (0=max diverse, 1=max relevance)
        "score_threshold": 0.0,        # min similarity score (0 = disabled)
        "show_sources": True,
        # Multi-file settings
        "multi_book_mode": False,      # enable querying multiple books at once
        "k_per_book": config.RETRIEVAL_K_PER_BOOK,
        "fusion_method": "rrf",        # "rrf" | "score" | "rerank"
        "rrf_k": 60,                   # RRF constant (higher = more weight to lower ranks)

        # Hybrid search
        "hybrid_search": config.HYBRID_SEARCH_ENABLED,
        "hybrid_alpha": config.HYBRID_ALPHA,
    },
}


KEY_ALIASES: Dict[str, str] = {
    "TOP_K": "retrieval_k",
    "SEARCH_TYPE": "search_type",
    "FUSION_METHOD": "fusion_method",
    "MMR_LAMBDA": "mmr_lambda",
    "SCORE_THRESHOLD": "score_threshold",
    "SHOW_SOURCES": "show_sources",
    "RERANK_ENABLED": "rerank_enabled",
    "RERANK_TOP_K": "rerank_top_k",
    "HYBRID_SEARCH": "hybrid_search",
    "HYBRID_ALPHA": "hybrid_alpha",
    "K_PER_BOOK": "k_per_book",
    "RRF_K": "rrf_k",
}


class SettingsManager:
    """Thread-safe (read-heavy) settings store backed by a JSON file."""

    def __init__(self) -> None:
        self._settings: Dict[str, Any] = {}
        self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        actual_key = KEY_ALIASES.get(key, key)
        key_lower = actual_key.lower()
        for section in ("rag", "model"):
            section_data = self._settings.get(section, {})
            if actual_key in section_data:
                value = section_data[actual_key]
                if actual_key != key:
                    logger.debug("get(%s) -> get(%s) = %s [%s]", key, actual_key, value, section)
                else:
                    logger.debug("get(%s) -> %s [%s]", key, value, section)
                return value
            for k, v in section_data.items():
                if k.lower() == key_lower:
                    if k != key:
                        logger.debug("get(%s) -> %s=%s [%s] (case-insensitive match)", key, k, v, section)
                    return v
        logger.debug("get(%s) -> default=%s", key, default)
        return default

    def get_settings(self) -> Dict[str, Any]:
        """Return a shallow copy of the current settings dict."""
        return {k: dict(v) if isinstance(v, dict) else v for k, v in self._settings.items()}

    def update_settings(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge *updates* into current settings, persist, and return new state."""
        self._settings = self._deep_merge(self._settings, updates)
        self._save()
        logger.info("Settings updated: %s", list(updates.keys()))
        return self.get_settings()

    def get_model_settings(self) -> Dict[str, Any]:
        return dict(self._settings.get("model", {}))

    def get_rag_settings(self) -> Dict[str, Any]:
        return dict(self._settings.get("rag", {}))

    @staticmethod
    def get_providers() -> Dict[str, Dict]:
        return PROVIDER_PRESETS

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
                    stored = json.load(fh)
                self._settings = self._deep_merge(DEFAULT_SETTINGS, stored)
                logger.info("Settings loaded from %s", SETTINGS_FILE)
                return
            except Exception as exc:
                logger.warning("Could not load settings (%s) — using defaults.", exc)
        self._settings = self._deep_merge({}, DEFAULT_SETTINGS)
        logger.info("Using default settings")

    def _save(self) -> None:
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
                json.dump(self._settings, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("Failed to save settings: %s", exc)

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        result = dict(base)
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = SettingsManager._deep_merge(result[key], val)
            else:
                result[key] = val
        return result