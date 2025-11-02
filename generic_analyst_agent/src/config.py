"""Configuration management.

Single Responsibility: Load and expose configuration values (API keys, etc.).
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the project-local .env (override any pre-set envs)
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
else:
    # Fallback to default search to support alternate setups
    load_dotenv(override=True)

# Exported constants (read once at import time, with lazy Streamlit secrets loading)
def _get_env_or_streamlit_secret(key: str) -> str | None:
    """Get value from env var first, then try Streamlit secrets as fallback."""
    value = os.getenv(key)
    if value:
        return value
    
    # Try Streamlit secrets if available (only in Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except (ImportError, FileNotFoundError, RuntimeError):
        pass
    
    return None

# LLM provider (Groq)
GROQ_API_KEY: str | None = _get_env_or_streamlit_secret("GROQ_API_KEY")

# Search provider (Google Custom Search JSON API)
GOOGLE_API_KEY: str | None = _get_env_or_streamlit_secret("GOOGLE_API_KEY")
GOOGLE_CSE_ID: str | None = _get_env_or_streamlit_secret("GOOGLE_CSE_ID")  # aka "cx"

# Google Gemini for summarization
GEMINI_API_KEY: str | None = _get_env_or_streamlit_secret("GEMINI_API_KEY")

# Optional: light validation/warnings without crashing import
if not GROQ_API_KEY:
    # Avoid noisy logs in libraries; keep a gentle hint for runtime.
    pass
if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    pass
if not GEMINI_API_KEY:
    pass
