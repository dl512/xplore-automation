"""
OpenAI-compatible API key and base URL for LangChain ChatOpenAI / OpenAI SDK.

Precedence:
1. If AI_GATEWAY_API_KEY is set → Vercel AI Gateway (Bearer key, OpenAI-compatible /v1).
  2. Else → OPENAI_API_KEY + OPENAI_BASE_URL or BASE_URL (e.g. OpenRouter).

Optional: AI_GATEWAY_BASE_URL (default https://ai-gateway.vercel.sh/v1).
"""

from __future__ import annotations

import os

DEFAULT_VERCEL_AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"


def get_openai_compatible_api_key() -> str:
    gateway_key = os.getenv("AI_GATEWAY_API_KEY", "").strip()
    if gateway_key:
        return gateway_key
    return (os.getenv("OPENAI_API_KEY") or "").strip()


def get_openai_compatible_base_url() -> str:
    if os.getenv("AI_GATEWAY_API_KEY", "").strip():
        custom = os.getenv("AI_GATEWAY_BASE_URL", "").strip()
        return custom or DEFAULT_VERCEL_AI_GATEWAY_BASE_URL
    return (
        os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("BASE_URL", "").strip()
    )


def get_openai_compatible_credentials() -> tuple[str, str]:
    return get_openai_compatible_api_key(), get_openai_compatible_base_url()
