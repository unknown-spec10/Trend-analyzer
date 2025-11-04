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
        self._llm: BaseChatModel = llm or ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3
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
        
        # Detect column types to give better guidance
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        instructions = f"""
You are an expert data analyst. Given a pandas DataFrame named `df`, write Python code that:

CRITICAL REQUIREMENTS:
1. Analyze the ACTUAL columns in the schema to understand the dataset domain
2. Answer the user's question using ONLY the data present in `df`
3. Build a STRUCTURED dictionary named `result` with these exact keys:
   {{"metric": str, "value": number, "period": str, "segment": str, "unit": str, "details": dict, "summary": str}}

FIELD SPECIFICATIONS:
- `metric`: Short name describing what was measured (e.g., "regional_distribution", "gender_breakdown")
- `value`: PRIMARY numeric answer (count, sum, average, percentage) - this is the MAIN result
- `period`: Time window if dates exist; otherwise "full_dataset" or describe the data scope
- `segment`: The primary grouping/category/demographic that answers the question
  * Inspect the user's question - if they ask about "region", use the region with highest value
  * If they ask about "gender" or "sex", use the dominant gender
  * If they ask about multiple dimensions, pick the MOST relevant one as segment
- `unit`: Appropriate unit ("records", "claims", "patients", "USD", "percentage", etc.)
- `details`: Dict with secondary breakdowns - MUST contain ACTUAL VALUES from the data
  * For multi-part questions, put the answer to the secondary part here
  * Example: If asking "which region claimed most AND which gender in that region", `details` should have {{"by_gender": {{"male": <number>, "female": <number>}}}}
  * NEVER use region names or other dimension values as keys in gender breakdowns
- `summary`: 1-2 sentence plain English answer to the question with ALL key numbers including secondary dimensions

DATASET CONTEXT (adapt your analysis to these columns):
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Date columns: {date_cols}

CODE RULES:
- Build a dict named `result` with ALL required keys (metric, value, period, segment, unit, details, summary)
- At the end, print ONLY: print(json.dumps(result))
- Do NOT import anything (json, pd, int, float, str, dict already available)
- Do NOT use print for anything else (no debug prints)
- Do NOT write comments, explanations, or markdown
- Use df.columns to verify column names before filtering
- Handle missing values gracefully with .dropna() or .fillna()
- Ensure `value` is numeric (use int() or float() to convert)
- Output ONLY executable Python code (no ```python fences, no explanations)

EXAMPLE CODE STRUCTURE:
# Find the answer using pandas
result_value = ...  # your calculation
result = {{
    "metric": "descriptive_name",
    "value": int(result_value),
    "period": "full_dataset",
    "segment": "category_name",
    "unit": "records",
    "details": {{}},
    "summary": "Plain English answer"
}}
print(json.dumps(result))

EXAMPLE FOR MULTI-DIMENSIONAL QUESTIONS:
# Question: "Which region has the most claims and which gender in that region has more?"
# Step 1: Find the region with highest claims
region_totals = df.groupby('region')['claim'].sum()
top_region = region_totals.idxmax()
top_region_value = int(region_totals.max())

# Step 2: Within that region, find gender breakdown
region_data = df[df['region'] == top_region]
gender_totals = region_data.groupby('gender')['claim'].sum()

# Step 3: Build result with BOTH answers
result = {{
    "metric": "regional_claim_analysis",
    "value": top_region_value,
    "period": "full_dataset",
    "segment": top_region,
    "unit": "USD",
    "details": {{
        "by_gender": gender_totals.to_dict(),
        "top_gender": gender_totals.idxmax(),
        "top_gender_amount": int(gender_totals.max())
    }},
    "summary": f"The {{top_region}} region has the highest claims ({{top_region_value}} USD). Within this region, {{gender_totals.idxmax()}} has claimed more ({{int(gender_totals.max())}} USD vs {{int(gender_totals.min())}} USD)."
}}
print(json.dumps(result))

User question:
{user_query}

DataFrame schema:
{schema_info}

Data sample (first 10 rows):
{head_str}
        """
        return textwrap.dedent(instructions).strip()

    def _run_query(self, user_query: str) -> Any:
        import logging
        logger = logging.getLogger(__name__)
        
        df = self._data_source.get_data()
        prompt = self._build_prompt(user_query, df)
        
        # Try up to 2 times if LLM fails to generate valid code
        max_attempts = 2
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"DataQueryTool attempt {attempt + 1}/{max_attempts}")
                resp = self._llm.invoke(prompt)  # type: ignore[attr-defined]
                code = _extract_code_blocks(getattr(resp, "content", str(resp)))
                code = self._sanitize_generated_code(code)
                
                logger.debug(f"Generated code:\n{code}")
                
                result = self._execute_code(code, df)
                
                # If result is a valid structured dict, return it
                if isinstance(result, dict) and "metric" in result:
                    logger.info("Successfully generated structured result")
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
        
        # If all attempts fail, use generic fallback
        logger.warning(f"All code generation attempts failed. Last error: {last_error}. Using fallback.")
        return self._fallback_fact(user_query, df)

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

    def _fallback_fact(self, user_query: str, df: pd.DataFrame) -> dict[str, Any]:
        """Generic fallback when LLM code generation fails.
        
        Provides basic dataset statistics without making assumptions about domain.
        """
        try:
            # Get basic info about the dataset
            total_rows = len(df)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Try to provide some basic insight
            summary_parts = [f"Dataset contains {total_rows:,} records"]
            
            # If there are categorical columns, find the most common category
            top_category = None
            if categorical_cols:
                first_cat_col = categorical_cols[0]
                value_counts = df[first_cat_col].value_counts()
                if not value_counts.empty:
                    top_category = str(value_counts.index[0])
                    top_count = int(value_counts.iloc[0])
                    summary_parts.append(f"Most common {first_cat_col}: {top_category} ({top_count} records)")
            
            # If there are numeric columns, provide a basic stat
            if numeric_cols:
                first_num_col = numeric_cols[0]
                total_sum = float(df[first_num_col].sum())
                summary_parts.append(f"Total {first_num_col}: {total_sum:,.2f}")
            
            summary = ". ".join(summary_parts)
            
            return {
                "metric": "dataset_overview",
                "value": total_rows,
                "period": "full_dataset",
                "segment": top_category or "all_records",
                "unit": "records",
                "summary": summary,
                "details": {
                    "numeric_columns": numeric_cols[:5],  # First 5
                    "categorical_columns": categorical_cols[:5],  # First 5
                    "note": "Fallback analysis - LLM code generation failed"
                }
            }
        except Exception as e:
            # Ultimate fallback if even basic stats fail
            return {
                "error": f"Could not analyze dataset: {e}",
                "metric": "unknown",
                "value": 0,
                "period": "unknown",
                "segment": "unknown",
                "unit": "unknown",
                "summary": "Unable to process the dataset.",
                "details": {"available_columns": list(df.columns)[:10]}
            }


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
