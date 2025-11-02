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

# Exported constants (read once at import time)
# LLM provider (Groq)
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

# Search provider (Google Custom Search JSON API)
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID: str | None = os.getenv("GOOGLE_CSE_ID")  # aka "cx"

# Google Gemini for summarization
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# Optional: light validation/warnings without crashing import
if not GROQ_API_KEY:
    # Avoid noisy logs in libraries; keep a gentle hint for runtime.
    pass
if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    pass
if not GEMINI_API_KEY:
    pass
