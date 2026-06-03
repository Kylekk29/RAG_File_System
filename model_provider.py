"""
model_provider.py — LLM abstraction with:
  • Runtime parameter overrides (temperature, top_p, …)
  • Streaming support via LangChain LCEL chain.stream()
  • Multi-provider via OpenAI-compatible endpoints (DeepSeek, Groq, Ollama, OpenAI…)
  • Robust retry logic with exponential back-off
"""

import json
import logging
import os
import re
import time
from typing import Dict, Generator, Iterator, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

import config
from settings_manager import SettingsManager

logger = logging.getLogger(__name__)


class ModelProvider:
    def __init__(self, settings_manager: Optional[SettingsManager] = None) -> None:
        logger.info("Initializing ModelProvider...")
        self.settings_mgr = settings_manager or SettingsManager()

        prompts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")
        try:
            with open(prompts_path, "r", encoding="utf-8") as fh:
                self.prompts: Dict = json.load(fh)
            logger.info("Loaded %d prompt templates", len(self.prompts))
        except FileNotFoundError:
            logger.warning("prompts.json not found — using empty prompt set")
            self.prompts = {}

        logger.info("ModelProvider ready")

    # ── Public: blocking ───────────────────────────────────────────────────────

    def chat(
        self,
        query: str,
        context: str = "",
        feature: str = "book_qa",
        runtime_params: Optional[Dict] = None,
        max_retries: int = 3,
    ) -> str:
        """Blocking call — returns the full response string."""
        logger.info("chat | feature=%s | query=%.80s…", feature, query)
        prompt_cfg = self._get_prompt(feature)
        if prompt_cfg is None:
            return f"Error: unknown prompt feature '{feature}'"

        chain = self._build_chain(prompt_cfg, runtime_params, streaming=False)
        inputs = {"context": context, "query": query}

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.time()
                response: str = chain.invoke(inputs)
                logger.info("chat | %d chars in %.2fs (attempt %d)", len(response), time.time() - t0, attempt)
                return response
            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc
                logger.warning("chat | connection/timeout attempt %d: %s", attempt, exc)
                self._wait(attempt, base=2, cap=30)
            except RateLimitError as exc:
                last_error = exc
                logger.warning("chat | rate-limit attempt %d: %s", attempt, exc)
                self._wait(attempt, base=5, cap=60)
            except APIError as exc:
                last_error = exc
                logger.error("chat | API error: %s", exc)
                break
            except Exception as exc:
                last_error = exc
                logger.error("chat | unexpected error attempt %d: %s", attempt, exc, exc_info=True)
                self._wait(attempt, base=2, cap=30)

        return f"AI service unavailable after {max_retries} attempts: {last_error}"

    # ── Public: streaming ──────────────────────────────────────────────────────

    def stream_chat(
        self,
        query: str,
        context: str = "",
        feature: str = "book_qa",
        runtime_params: Optional[Dict] = None,
    ) -> Iterator[str]:
        """Generator — yields text chunks as they arrive from the LLM."""
        logger.info("stream_chat | feature=%s | query=%.80s…", feature, query)
        prompt_cfg = self._get_prompt(feature)
        if prompt_cfg is None:
            yield f"Error: unknown prompt feature '{feature}'"
            return

        chain = self._build_chain(prompt_cfg, runtime_params, streaming=True)
        inputs = {"context": context, "query": query}
        try:
            for chunk in chain.stream(inputs):
                yield chunk
        except (APIConnectionError, APITimeoutError) as exc:
            logger.error("stream_chat | connection error: %s", exc)
            yield f"\n\n[Connection error: {exc}]"
        except RateLimitError as exc:
            logger.error("stream_chat | rate limit: %s", exc)
            yield f"\n\n[Rate limit reached — please retry shortly]"
        except Exception as exc:
            logger.error("stream_chat | error: %s", exc, exc_info=True)
            yield f"\n\n[Error: {exc}]"

    # ── Compatibility: generate_response (delegates to chat) ──────────────────

    def generate_response(self, feature: str, context: str = "", query: str = "", runtime_params: Optional[Dict] = None) -> str:
        logger.info("generate_response: feature=%s, query=%.60s, context_len=%d, runtime_params=%s",
                     feature, query, len(context), runtime_params)
        return self.chat(query=query, context=context, feature=feature, runtime_params=runtime_params)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_prompt(self, feature: str) -> Optional[Dict]:
        cfg = self.prompts.get(feature)
        if cfg is None:
            logger.error("Unknown prompt feature: %s", feature)
        return cfg

    def _build_chain(
        self,
        prompt_cfg: Dict,
        runtime_params: Optional[Dict],
        streaming: bool,
    ):
        """Build a LangChain LCEL chain with effective LLM parameters."""
        llm = self._create_llm(runtime_params or {}, streaming=streaming)
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_cfg["system_prompt"]),
            ("user", prompt_cfg["user_template"]),
        ])
        return prompt | llm | StrOutputParser()

    def _create_llm(self, overrides: Dict, streaming: bool = False) -> ChatOpenAI:
        """Instantiate ChatOpenAI with merged settings + per-request overrides."""
        s = self.settings_mgr.get_model_settings()
        effective_api_key = overrides.get("api_key") or config.API_KEY

        kwargs: Dict = {
            "model": overrides.get("model_name") or s.get("model_name") or config.MODEL_NORMAL,
            "openai_api_key": effective_api_key,
            "openai_api_base": overrides.get("base_url") or s.get("base_url") or config.BASE_URL,
            "temperature": overrides.get("temperature", s.get("temperature", 0.3)),
            "top_p": overrides.get("top_p", s.get("top_p", 0.95)),
            "timeout": s.get("timeout", config.TIMEOUT),
            "max_retries": s.get("max_retries", config.MAX_RETRIES),
            "streaming": streaming,
        }

        # Optional params — only add if non-zero to avoid API rejection
        presence = overrides.get("presence_penalty", s.get("presence_penalty", 0.0))
        frequency = overrides.get("frequency_penalty", s.get("frequency_penalty", 0.0))
        if presence != 0.0:
            kwargs["presence_penalty"] = presence
        if frequency != 0.0:
            kwargs["frequency_penalty"] = frequency

        logger.debug(
            "_create_llm | model=%s base=%s temp=%.2f top_p=%.2f streaming=%s",
            kwargs["model"], kwargs["openai_api_base"],
            kwargs["temperature"], kwargs["top_p"],
            streaming,
        )
        return ChatOpenAI(**kwargs)

    @staticmethod
    def _wait(attempt: int, base: int = 2, cap: int = 30) -> None:
        wait = min(base ** attempt, cap)
        logger.info("  Retrying in %ds…", wait)
        time.sleep(wait)

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def extract_json(text: str) -> Dict:
        """Best-effort JSON extraction from a free-text LLM response."""
        cleaned = re.sub(r"^```(?:json)?\s*|```\s*$", "", text.strip(), flags=re.MULTILINE).strip()
        for pattern, flags in [
            (r"\{.*\}", re.DOTALL),
            (r"\[.*\]", re.DOTALL),
        ]:
            for candidate in [cleaned, cleaned.replace("'", '"')]:
                m = re.search(pattern, candidate, flags)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except json.JSONDecodeError:
                        pass
        logger.warning("extract_json: could not parse response")
        return {"error": "Failed to parse AI response", "raw": text[:800]}