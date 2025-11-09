from typing import Any
import logging

logger = logging.getLogger(__name__)

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None  # type: ignore

# Local import delayed to avoid circular issues
from . import config


def _invoke_gemini_text(prompt: Any) -> Any:
    """Fallback invocation using Google Gemini; returns LangChain-compatible response."""
    try:
        import google.generativeai as genai  # type: ignore
        from langchain_core.messages import AIMessage
    except Exception as e:
        raise RuntimeError("Gemini client not available for fallback") from e

    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured for fallback")

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Convert prompt to string format for Gemini
    prompt_text = str(prompt)
    if isinstance(prompt, list):
        # Handle LangChain message format
        prompt_text = "\n".join([
            str(msg.get("content", msg)) if isinstance(msg, dict) 
            else str(getattr(msg, "content", msg))
            for msg in prompt
        ])
    
    # Use generate_content to match SDK usage in tools.py
    resp = model.generate_content(prompt_text)
    
    # Try .text attribute first (most common)
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        # Return LangChain-compatible AIMessage
        return AIMessage(content=text.strip())
    
    # Some SDKs return candidates list
    cand = getattr(resp, "candidates", None)
    if isinstance(cand, list) and cand:
        first = cand[0]
        part_text = getattr(first, "content", None)
        if isinstance(part_text, str) and part_text.strip():
            return AIMessage(content=part_text.strip())
    
    # Some SDK variants use .output
    out = getattr(resp, "output", None)
    if isinstance(out, list) and out:
        part = getattr(out[0], "content", None)
        if isinstance(part, str) and part.strip():
            return AIMessage(content=part.strip())
    
    raise RuntimeError("Gemini fallback returned no text")


class GroqWithGeminiFallback:
    """Wrapper around ChatGroq that falls back to Gemini immediately on rate limits.

    It preserves the .invoke(...) method used throughout the codebase.
    """

    def __init__(self, *args, gemini_fn=None, **kwargs):
        if ChatGroq is None:
            raise RuntimeError("ChatGroq is not importable in this environment")
        self._groq = ChatGroq(*args, **kwargs)
        self._gemini_fn = gemini_fn or _invoke_gemini_text

    def invoke(self, prompt: Any) -> Any:
        try:
            return self._groq.invoke(prompt)
        except Exception as e:
            error_msg = str(e)
            msg = error_msg.lower()
            # Immediate fallback on rate-limit indicators
            if "rate limit" in msg or "429" in msg or "tokens per day" in msg or "rate_limit_exceeded" in msg:
                logger.warning(f"Groq rate limit detected: {error_msg[:200]}")
                logger.info("Switching to Gemini fallback immediately (no retries).")
                try:
                    result = self._gemini_fn(prompt)
                    logger.info("Gemini fallback succeeded.")
                    return result
                except Exception as gemini_error:
                    logger.error(f"Gemini fallback failed: {gemini_error}")
                    raise RuntimeError(f"Both Groq and Gemini failed. Groq: {error_msg[:100]}, Gemini: {str(gemini_error)[:100]}")
            # Otherwise re-raise
            raise

    def __getattr__(self, name: str) -> Any:
        # Forward other attributes/methods to the underlying model
        return getattr(self._groq, name)