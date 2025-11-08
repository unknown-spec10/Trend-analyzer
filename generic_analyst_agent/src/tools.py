"""Agent tools.

- DataQueryTool: schema-aware data query tool powered by an internal LLM that emits pandas code.
- search_for_probable_causes: web search summarizer using Tavily.

Notes on safety: This demo uses `exec` to run generated code. Do NOT do this in production
without strong sandboxing, resource limits, and allow-listing. Treat generated code as untrusted.
"""
from __future__ import annotations

import io
import re
import textwrap
from contextlib import redirect_stdout
from typing import Any, Callable
import json
import ast

import pandas as pd
from langchain.tools import tool
from langchain_groq import ChatGroq

from .prompts import get_data_analysis_prompt
from langchain_core.language_models.chat_models import BaseChatModel

from .data_source import BaseDataSource
from . import config


def _extract_code_blocks(s: str) -> str:
    """Return first python code block or the raw string if no fences.

    Supports ```python ...``` or ``` ...``` fences.
    """
    fence = re.compile(r"```(?:(?:python)|(?:py))?\n(.*?)```", re.DOTALL | re.IGNORECASE)
    m = fence.search(s)
    return m.group(1).strip() if m else s.strip()


class DataQueryTool:
    """Generic, schema-aware data query tool.

    - Accepts a BaseDataSource (DIP)
    - Inspects the schema via df.info() and df.head()
    - Uses an internal LLM to produce safe pandas code that prints the answer
    - Exposes a public Tool on the instance: `query_data`
    """

    def __init__(self, data_source: BaseDataSource, llm: BaseChatModel | None = None) -> None:
        self._data_source = data_source
        # Instantiate Groq chat model; ignore type checker arg mismatch if stubs differ.
        # temperature=0.2 balances determinism with functionality
        # (0.0-0.1 too conservative and causes errors, 0.3+ introduces too much randomness)
        self._llm: BaseChatModel = llm or ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2
        )  # type: ignore[arg-type]

        # Create a Tool bound to this instance using a closure so we can pass it to the agent
        @tool("query_data")
        def query_data(query: str) -> str:
            """Query the local tabular data to obtain a precise, quantitative fact.

            Always return a single, crisp fact with units or counts when relevant. Use the
            current dataset only (no external knowledge). If the question is about a period
            or location, group or filter accordingly and report the exact figure you found.
            """
            return self._run_query(query)

        # public attribute used by the agent
        self.query_data = query_data  # type: ignore[attr-defined]

    # -------- internal methods --------
    def _build_prompt(self, user_query: str, df: pd.DataFrame) -> str:
        # capture df.info() output
        buf = io.StringIO()
        df.info(buf=buf)
        schema_info = buf.getvalue()
        head_str = df.head(10).to_string(index=False)

        # Try to parse likely date columns to datetime
        likely_date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'timestamp'])]
        for col in likely_date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass

        # Detect column types to give better guidance
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

        # Use centralized prompt
        return get_data_analysis_prompt(
            user_query=user_query,
            schema_info=schema_info,
            head_str=head_str,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            date_cols=date_cols
        )

    def _run_query(self, user_query: str) -> Any:
        import logging
        logger = logging.getLogger(__name__)
        
        df = self._data_source.get_data()

        # Validation layer: analyze question tokens vs dataset columns
        validation_error = self._validate_question(user_query, df)
        if validation_error is not None:
            logger.info(f"Question validation blocked execution: {validation_error['details']}")
            validation_error["generated_code"] = None
            try:
                validation_error["internal_context"] = self._build_internal_context(df)
            except Exception:
                pass
            return validation_error
        prompt = self._build_prompt(user_query, df)
        
        # Try up to 2 times if LLM fails to generate valid code
        max_attempts = 2
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"DataQueryTool attempt {attempt + 1}/{max_attempts}")
                resp = self._llm.invoke(prompt)  # type: ignore[attr-defined]
                logger.info(f"Raw LLM response: {getattr(resp, 'content', str(resp))}")
                code = _extract_code_blocks(getattr(resp, "content", str(resp)))
                code = self._sanitize_generated_code(code)

                logger.debug(f"Generated code:\n{code}")

                result = self._execute_code(code, df)

                # If result is a valid structured dict, attach generated code for UI visibility
                if isinstance(result, dict) and "metric" in result:
                    logger.info("Successfully generated structured result")
                    # Only attach code if not already present
                    if "generated_code" not in result:
                        result["generated_code"] = code.strip()
                    # Attach core pandas query lines for UI clarity
                    try:
                        result["core_code"] = self._extract_core_pandas(code)
                    except Exception:
                        result["core_code"] = None
                    # Attach internal context summary used to build the prompt
                    try:
                        ctx = self._build_internal_context(df)
                        result["internal_context"] = ctx
                    except Exception:
                        pass
                    return result

                # Log what we got instead
                logger.warning(f"Invalid result type or missing 'metric': {type(result)}")
                last_error = f"Result was {type(result).__name__}, not a valid dict with 'metric' key. Got: {str(result)[:200]}"

                # If we got an error message, try again
                if isinstance(result, str) and "Error" in result and attempt < max_attempts - 1:
                    logger.info(f"Retrying due to error: {result[:100]}")
                    prompt += f"\n\nPREVIOUS ATTEMPT FAILED:\n{result}\n\nPlease fix the code and try again."
                    continue

            except Exception as e:
                logger.warning(f"Exception in attempt {attempt + 1}: {e}")
                last_error = str(e)
                if attempt < max_attempts - 1:
                    prompt += f"\n\nPREVIOUS ATTEMPT FAILED:\n{str(e)}\n\nPlease fix the code and try again."
                    continue
        
        # If primary model fails, attempt Gemini fallback LLM for code generation
        try:
            logger.info("Primary LLM failed; attempting Gemini fallback for code generation.")
            gemini_resp = self._invoke_gemini(prompt)
            logger.info(f"Raw Gemini response: {gemini_resp}")
            gcode = _extract_code_blocks(gemini_resp)
            gcode = self._sanitize_generated_code(gcode)
            logger.debug(f"Gemini generated code:\n{gcode}")
            gresult = self._execute_code(gcode, df)
            if isinstance(gresult, dict) and "metric" in gresult:
                if "generated_code" not in gresult:
                    gresult["generated_code"] = gcode.strip()
                try:
                    gresult["core_code"] = self._extract_core_pandas(gcode)
                except Exception:
                    gresult["core_code"] = None
                try:
                    gresult["internal_context"] = self._build_internal_context(df)
                except Exception:
                    pass
                return gresult
            last_error = f"Gemini fallback returned invalid result: {type(gresult)} — {str(gresult)[:200]}"
        except Exception as e:
            last_error = f"Gemini fallback failed: {e}"

        # Both models failed: return structured error (no rule-based data fallback)
        logger.warning(f"All code generation attempts failed. Last error: {last_error}.")
        err = {
            "metric": "error",
            "value": None,
            "period": "unknown",
            "segment": "unknown",
            "unit": "unknown",
            "summary": "Code generation failed after trying primary and fallback LLMs",
            "details": {"last_error": last_error},
        }
        try:
            err["internal_context"] = self._build_internal_context(df)
        except Exception:
            pass
        err["generated_code"] = None
        return err

    def _build_internal_context(self, df: pd.DataFrame) -> dict[str, Any]:
        """Summarize schema and sample for display alongside results."""
        buf = io.StringIO()
        df.info(buf=buf)
        schema_info = buf.getvalue()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        try:
            head_sample = df.head(3).to_dict(orient="records")
        except Exception:
            head_sample = []
        return {
            "columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "date_columns": date_cols,
            "schema_info": schema_info,
            "head_sample": head_sample,
        }

    def _extract_core_pandas(self, code: str) -> str | None:
        """Extract the core pandas operations involving df, excluding result packaging/printing.

        Heuristics: keep lines before the first 'result =' that reference df and contain common
        pandas operations, plus their simple assignments.
        """
        ops = ("groupby", "agg", "sum(", "mean(", "count(", "min(", "max(", "query(",
               "loc[", "iloc[", "sort_values", "pivot_table", "value_counts", "merge(",
               "assign(", "resample(", "rolling(")
        lines = code.splitlines()
        core: list[str] = []
        for line in lines:
            if line.strip().startswith("result ") or line.strip().startswith("result="):
                break
            if "df" in line and any(op in line for op in ops):
                core.append(line.rstrip())
            elif "df" in line and ("=" in line):
                # keep simple assignments involving df
                core.append(line.rstrip())
        # Remove trivial prints or json
        core = [ln for ln in core if ("print(" not in ln and "json.dumps" not in ln)]
        return "\n".join(core).strip() or None


    def _invoke_gemini(self, prompt: str) -> str:
        """Invoke Gemini model with the same prompt to get alternate code.

        Returns the raw text content from Gemini. Raises on import/key issues to propagate error context.
        """
        import google.generativeai as genai  # type: ignore
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not configured")
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        # Some SDKs return candidates list
        cand = getattr(resp, "candidates", None)
        if isinstance(cand, list) and cand:
            # Try to extract first text segment
            first = cand[0]
            part_text = getattr(first, "content", None)
            if isinstance(part_text, str) and part_text.strip():
                return part_text
        raise RuntimeError("Gemini response did not contain text content")

    def _validate_question(self, user_query: str, df: pd.DataFrame) -> dict[str, Any] | None:
        """Validate the question against available dataset columns.

        Logic:
        1. Tokenize question (simple split on non-alphanumerics) to extract potential column references.
        2. Score matches: exact column name, underscore/space-insensitive match, singular/plural match.
        3. If the question references domain-specific column keywords that do NOT exist, and no valid matches are found,
           return structured error.
        4. If no column-like tokens found at all (purely generic question) allow LLM (it may do overall stats).
        5. Temporal tokens (month/quarter/year) require at least one date/time column; else error.
        """
        import re
        q = user_query.lower()
        col_map = {c.lower(): c for c in df.columns}
        cols_lower = set(col_map.keys())

        # Basic tokenization
        tokens = [t for t in re.split(r"[^a-zA-Z0-9_]+", q) if t]
        if not tokens:
            return None  # nothing to validate

        # Identify temporal tokens
        temporal_tokens = (
            {"q1","q2","q3","q4","quarter","quarters","month","months","year","years"}
            | {"january","february","march","april","may","june","july","august","september","october","november","december",
               "jan","feb","mar","apr","jun","jul","aug","sep","sept","oct","nov","dec"}
        )
        has_temporal = any(t in temporal_tokens for t in tokens)
        has_date_col = any(str(dtype).startswith("datetime") for dtype in df.dtypes) or any(
            any(k in c.lower() for k in ["date","time","timestamp","datetime","dt"]) for c in df.columns)
        if has_temporal and not has_date_col:
            return {
                "metric": "error",
                "value": None,
                "period": "unknown",
                "segment": "unknown",
                "unit": "unknown",
                "summary": "Temporal analysis requested but dataset has no date/time columns",
                "details": {"error": "Missing date/time columns", "question": user_query}
            }

        # Domain keywords which should map to columns
        domain_keywords = {
            "region","claim","salary","department","product","revenue","quantity","price","temperature","humidity",
            "age","gender","smoker","diabetic","bmi","bloodpressure"
        }
        mentioned_domain = {t for t in tokens if t in domain_keywords}
        matched_columns: set[str] = set()
        missing_domain: set[str] = set()

        # Matching logic
        for t in mentioned_domain:
            if t in cols_lower:
                matched_columns.add(col_map[t])
            else:
                # try singular/plural heuristics
                singular = t[:-1] if t.endswith("s") else t
                plural = t + "s" if not t.endswith("s") else t[:-1]
                if singular in cols_lower:
                    matched_columns.add(col_map[singular])
                elif plural in cols_lower:
                    matched_columns.add(col_map[plural])
                else:
                    # try substring match (e.g., "temperature" matches "temperature_celsius")
                    substring_match = any(t in col or col.startswith(t) for col in cols_lower)
                    if substring_match:
                        # Find the matching column
                        for col_lower, col_orig in col_map.items():
                            if t in col_lower or col_lower.startswith(t):
                                matched_columns.add(col_orig)
                                break
                    else:
                        missing_domain.add(t)

        # If domain terms mentioned but none matched
        if mentioned_domain and not matched_columns and missing_domain == mentioned_domain:
            return {
                "metric": "error",
                "value": None,
                "period": "unknown",
                "segment": "unknown",
                "unit": "unknown",
                "summary": "Question references columns not present in dataset",
                "details": {"missing_columns": sorted(missing_domain), "question": user_query}
            }

        return None

    def _sanitize_generated_code(self, code: str) -> str:
        """Remove unsafe or disallowed constructs from model-generated code.

        - Strip markdown fences
        - Remove any import statements (e.g., `import pandas as pd`, `from x import y`)
        """
        # Remove code fences preemptively
        cleaned = re.sub(r"```[\s\S]*?```", "", code)
        # Drop import lines entirely
        lines = []
        for line in cleaned.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def _execute_code(self, code: str, df: pd.DataFrame) -> Any:
        """Execute generated code in a sandboxed environment with resource limits.

        Uses subprocess isolation for production safety.
        Returns a dict when the printed output is JSON or a Python dict; otherwise returns a string message.
        """
        try:
            from .sandbox import get_executor
            use_sandbox = True
        except ImportError:
            use_sandbox = False
            import logging
            logging.getLogger(__name__).warning("Sandbox module not available. Using fallback execution.")
        
        if use_sandbox:
            # Use sandboxed subprocess execution (RECOMMENDED)
            executor = get_executor(timeout_seconds=30, memory_limit_mb=512)
            result = executor.execute(code, df)
            
            if "error" in result:
                return f"Error executing generated code: {result['error']}\nCode:\n{code}"
            
            printed = result.get("output", "").strip()
        else:
            # Fallback to in-process execution (LESS SAFE)
            local_vars: dict[str, Any] = {"df": df, "pd": pd, "json": json}
            stdout = io.StringIO()
            try:
                with redirect_stdout(stdout):
                    exec(
                        code,
                        {
                            "__builtins__": {
                                # Essential Python functions
                                "len": len,
                                "range": range,
                                "min": min,
                                "max": max,
                                "sum": sum,
                                "all": all,
                                "any": any,
                                "int": int,
                                "float": float,
                                "str": str,
                                "dict": dict,
                                "list": list,
                                "tuple": tuple,
                                "set": set,
                                "abs": abs,
                                "round": round,
                                "sorted": sorted,
                                "enumerate": enumerate,
                                "zip": zip,
                                "print": print,
                                "bool": bool,
                                "isinstance": isinstance,
                                "type": type,
                            }
                        },
                        local_vars,
                    )
            except Exception as e:
                return f"Error executing generated code: {e}\nCode:\n{code}"

            printed = stdout.getvalue().strip()
            # If author forgot to print, try to read `result`
            if not printed and "result" in local_vars:
                try:
                    return dict(local_vars["result"])  # type: ignore[arg-type]
                except Exception:
                    printed = str(local_vars["result"])

        if not printed:
            return "No output produced by the generated code."

        # Try to parse as JSON
        try:
            parsed = json.loads(printed)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # Try to parse as Python literal dict
        try:
            lit = ast.literal_eval(printed)
            if isinstance(lit, dict):
                return lit
        except Exception:
            pass

        return printed


@tool("search_for_probable_causes")
def search_for_probable_causes(payload: Any) -> str:
    """Search the web for likely real-world causes of a specific, factual observation.

    Input payload may be:
    - A plain string query (backward compatible), or
    - A JSON string or dict with keys: {
        "query": str,
        "dateRestrict": str | None,  # e.g., "d30", "m6", "y1"
        "negative_keywords": list[str] | None
      }

    Provide a concise synthesis (3-6 sentences) pulling from multiple sources. Prefer
    causal explanations, industry reports, and news analysis over vendor pages. Avoid
    raw link dumps; instead, summarize key points and cite sources inline (title + URL).
    """
    try:
        from googleapiclient.discovery import build  # type: ignore
    except Exception as e:
        return f"Google API client not available: {e}"

    if not config.GOOGLE_API_KEY or not config.GOOGLE_CSE_ID:
        return "Google Search configuration missing (GOOGLE_API_KEY/GOOGLE_CSE_ID)."

    # Normalize payload
    query: str = ""
    date_restrict: str | None = None
    neg_keywords: list[str] = []
    try:
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    query = str(parsed.get("query", ""))
                    date_restrict = parsed.get("dateRestrict") or None
                    neg_keywords = list(parsed.get("negative_keywords") or [])
                else:
                    query = payload
            except Exception:
                query = payload
        elif isinstance(payload, dict):
            query = str(payload.get("query", ""))
            date_restrict = payload.get("dateRestrict") or None
            neg_keywords = list(payload.get("negative_keywords") or [])
        else:
            query = str(payload)
    except Exception:
        query = str(payload)

    # Apply negative keywords inline in the query
    neg_part = " ".join(f"-{kw}" for kw in neg_keywords if kw)
    full_query = f"{query} {neg_part}".strip()

    try:
        from googleapiclient.errors import HttpError  # type: ignore
        service = build("customsearch", "v1", developerKey=config.GOOGLE_API_KEY)
        params = {"q": full_query, "cx": config.GOOGLE_CSE_ID, "num": 5}
        if date_restrict:
            params["dateRestrict"] = date_restrict
        resp = service.cse().list(**params).execute()
    except HttpError as e:  # type: ignore
        # Provide clearer guidance for common 403 enablement issues
        msg = str(e)
        if "accessNotConfigured" in msg or "has not been used in project" in msg:
            return (
                "Google Custom Search API not enabled for this project. Please: "
                "1) Open https://console.developers.google.com/apis/library/customsearch.googleapis.com, "
                "2) Select your project, 3) Click Enable, 4) Retry in a few minutes."
            )
        return f"Search failed: {e}"
    except Exception as e:
        return f"Search failed: {e}"

    items = (resp or {}).get("items", [])
    if not items:
        return "No search results found."

    # Prepare context for Gemini summarization
    sources = []
    for item in items[:5]:
        sources.append({
            "title": (item.get("title") or "Untitled").strip(),
            "url": (item.get("link") or "").strip(),
            "snippet": re.sub(r"\s+", " ", (item.get("snippet") or "").strip()),
        })

    # Use Gemini 2.5 Flash to synthesize a concise summary with inline citations
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:
        # Fallback: return raw sources when Gemini client not available
        lines = ["Top findings:"] + [f"- {s['title']} — {s['snippet']} ({s['url']})" for s in sources]
        return json.dumps({
            "summary": "\n".join(lines),
            "sources": sources
        })

    if not config.GEMINI_API_KEY:
        # Fallback: return raw sources if key missing
        lines = ["Top findings (GEMINI_API_KEY missing):"] + [f"- {s['title']} — {s['snippet']} ({s['url']})" for s in sources]
        return json.dumps({
            "summary": "\n".join(lines),
            "sources": sources
        })

    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        context_lines: list[str] = []
        for i, s in enumerate(sources, start=1):
            context_lines.append(f"{i}. {s['title']}\nURL: {s['url']}\nSnippet: {s['snippet']}")
        context_text = "\n".join(context_lines)
        prompt = (
            "You are a root-cause analyst. Given the user topic and retrieved web snippets, "
            "write a concise synthesis (3-6 sentences) explaining likely causes. Cite sources inline "
            "with (Title - URL). Avoid speculation; prefer causal mechanisms and data-backed explanations.\n\n"
            f"User topic: {full_query}\n\nSources:\n{context_text}"
        )
        result = model.generate_content(prompt)
        text = getattr(result, "text", None) or getattr(result, "candidates", None)
        if isinstance(text, str) and text.strip():
            # Return JSON with both summary and sources
            return json.dumps({
                "summary": text.strip(),
                "sources": sources
            })
        # Fallback to raw sources if unexpected response
        lines = ["Top findings:"] + [f"- {s['title']} — {s['snippet']} ({s['url']})" for s in sources]
        return json.dumps({
            "summary": "\n".join(lines),
            "sources": sources
        })
    except Exception as e:
        # On error, fall back to listing search results
        lines = [f"Synthesis failed: {e}", "Top findings:"] + [f"- {s['title']} — {s['snippet']} ({s['url']})" for s in sources]
        return json.dumps({
            "summary": "\n".join(lines),
            "sources": sources
        })


@tool("summarize_text")
def summarize_text(text: str) -> str:
    """Summarize the given text into 3-6 concise sentences highlighting the key points."""
    if not text or not text.strip():
        return "No text provided to summarize."
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3
        )  # type: ignore[arg-type]
        prompt = (
            "Summarize the following content into 3-6 concise sentences focusing on the most important facts, "
            "trends, and causal drivers. Avoid fluff.\n\nCONTENT:\n" + text.strip()
        )
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp)).strip()
    except Exception as e:
        return f"Summarization failed: {e}"
