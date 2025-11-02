"""Configuration management.

Single Responsibility: Load and expose configuration values (API keys, etc.).

This module supports both local and cloud environments:
- Local: Reads from .env file via python-dotenv
- Streamlit Cloud: Reads from st.secrets (configured in Streamlit Cloud UI)
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Detect environment: Local or Streamlit Cloud
def _is_streamlit_cloud() -> bool:
    """Check if running in Streamlit Cloud environment."""
    try:
        import streamlit as st
        return hasattr(st, "secrets") and len(st.secrets) > 0
    except (ImportError, FileNotFoundError, RuntimeError):
        return False

# Load environment variables from .env file (for local development)
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
    print(f"[Config] Loaded secrets from local .env file: {_ENV_PATH}")
else:
    # Fallback to default search
    load_dotenv(override=True)

def _get_env_or_streamlit_secret(key: str) -> str | None:
    """
    Get configuration value with fallback chain:
    1. Environment variable (set by .env in local, or system env)
    2. Streamlit secrets (if running in Streamlit Cloud)
    
    Returns None if key not found in either location.
    """
    # First priority: Environment variables (works in both local and cloud)
    value = os.getenv(key)
    if value:
        return value
    
    # Second priority: Streamlit Cloud secrets (only available in cloud)
    if _is_streamlit_cloud():
        try:
            import streamlit as st
            if key in st.secrets:
                return st.secrets[key]
        except (ImportError, FileNotFoundError, RuntimeError, KeyError):
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
