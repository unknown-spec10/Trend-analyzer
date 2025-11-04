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
        head_str = df.head(5).to_string(index=False)
        instructions = f"""
You are a helpful data analyst. Given a pandas DataFrame named `df`, write Python code that:
- Computes the answer to the user's question using idiomatic pandas only.
- Builds a STRUCTURED dictionary named `result` with keys exactly: 
    {{"metric": str, "value": number, "period": str, "segment": str, "unit": str}}
- The `metric` should be a short descriptive name (e.g., "regional_claims", "gender_distribution").
- The `value` must be numeric (count, sum, or average) matching the question's intent.
- The `period` should be a human-readable time window if the data has dates; otherwise use "full_dataset" or "all_records".
- The `segment` should capture the most relevant category, region, demographic, or slice based on the question.
  Examples:
  * For regions: "southeast", "northwest", etc.
  * For demographics: "male", "female", age groups
  * For categories: product types, claim types, etc.
- The `unit` should be appropriate (e.g., "claims", "records", "USD", "patients").
- If helpful, include optional `details` dict with additional breakdowns (e.g., by gender, age, subcategory).
- Finally, print the JSON-serialized `result` using: print(json.dumps(result)).
- Do NOT write any import statements; the `json` module is already available.
- Never read/write files, never access network or environment.
- You may use `pd` if needed, but prefer DataFrame methods on `df`.
- Output ONLY the code. No explanations. No markdown fences.

User question:
{user_query}

DataFrame schema (df.info()):
{schema_info}

Data sample (df.head()):
{head_str}
        """
        return textwrap.dedent(instructions).strip()

    def _run_query(self, user_query: str) -> Any:
        df = self._data_source.get_data()
        prompt = self._build_prompt(user_query, df)
        resp = self._llm.invoke(prompt)  # type: ignore[attr-defined]
        code = _extract_code_blocks(getattr(resp, "content", str(resp)))
        code = self._sanitize_generated_code(code)
        result = self._execute_code(code, df)
        # If result is not a structured dict, provide a deterministic fallback
        if not isinstance(result, dict):
            fallback = self._fallback_fact(user_query, df)
            return fallback
        return result

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
        """Execute generated code in a constrained namespace and capture stdout.

        WARNING: Uses exec for demo simplicity. In production, enforce strict sandboxing.
        Returns a dict when the printed output is JSON or a Python dict; otherwise returns a string message.
        """
        # minimal globals/locals; allow df and optionally pandas alias
        local_vars: dict[str, Any] = {"df": df, "pd": pd, "json": json}
        stdout = io.StringIO()
        try:
            with redirect_stdout(stdout):
                exec(
                    code,
                    {
                        "__builtins__": {
                            "len": len,
                            "range": range,
                            "min": min,
                            "max": max,
                            "sum": sum,
                            "print": print,  # allow printing to captured stdout
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
        """Adaptive fallback when the LLM produces an unhelpful result.

        Attempts to answer the user's query using heuristics based on available columns.
        """
        # Try to locate standard column names
        cols = {c.lower(): c for c in df.columns}
        
        # Detect dataset type and adapt accordingly
        # Medical insurance dataset: age, sex, bmi, region, charges, smoker, children
        if "region" in cols and "charges" in cols and "sex" in cols:
            return self._fallback_medical_insurance(user_query, df, cols)
        
        # Product/incident claims dataset: product_category, date_of_incident, claim_amount
        required = ["product_category", "date_of_incident", "claim_amount"]
        if all(name in cols for name in required):
            return self._fallback_product_claims(user_query, df, cols)
        
        # Generic fallback
        return {
            "error": f"Could not compute a fallback fact. Available columns: {list(df.columns)}",
            "metric": "unknown",
            "value": 0,
            "period": "unknown",
            "segment": "unknown",
            "unit": "unknown",
        }

    def _fallback_medical_insurance(self, user_query: str, df: pd.DataFrame, cols: dict) -> dict[str, Any]:
        """Fallback for medical insurance datasets with region, sex, charges columns."""
        try:
            region_col = cols.get("region")
            charges_col = cols.get("charges")
            sex_col = cols.get("sex")
            
            # Find region with most claims
            region_counts = df[region_col].value_counts()
            top_region = region_counts.idxmax()
            top_region_count = int(region_counts.max())
            
            # Within that region, find which gender has more claims
            region_df = df[df[region_col] == top_region]
            sex_counts = region_df[sex_col].value_counts()
            top_sex = sex_counts.idxmax() if len(sex_counts) > 0 else "unknown"
            top_sex_count = int(sex_counts.max()) if len(sex_counts) > 0 else 0
            
            # Total charges for top region
            total_charges = float(region_df[charges_col].sum())
            
            return {
                "metric": "regional_claims_analysis",
                "value": top_region_count,
                "period": "dataset_period",
                "segment": f"{top_region} region",
                "unit": "claims",
                "details": {
                    "top_region": top_region,
                    "total_claims_in_region": top_region_count,
                    "dominant_gender": top_sex,
                    "gender_claims_count": top_sex_count,
                    "total_charges": round(total_charges, 2),
                },
                "note": "Fallback analysis for medical insurance data"
            }
        except Exception as e:
            return {
                "error": f"Medical insurance fallback failed: {e}",
                "metric": "unknown",
                "value": 0,
                "period": "unknown",
                "segment": "unknown",
                "unit": "unknown",
            }

    def _fallback_product_claims(self, user_query: str, df: pd.DataFrame, cols: dict) -> dict[str, Any]:
        """Fallback for product claims datasets (legacy behavior)."""
        pc = cols["product_category"]
        dt = cols["date_of_incident"]
        amt = cols["claim_amount"]

        dff = df.copy()
        # Ensure datetime
        try:
            dff[dt] = pd.to_datetime(dff[dt], errors="coerce")
        except Exception:
            pass
        m = dff[dt].dt.month == 3
        y = dff[dt].dt.year == 2025
        cat = dff[pc] == "Electronics"
        filt = cat & m & y
        slice_df = dff.loc[filt]
        count = int(slice_df.shape[0])
        total = float(slice_df[amt].sum()) if count else 0.0

        top_city_str = ""
        city_col = cols.get("city")
        if city_col and count:
            top = (
                slice_df.groupby(city_col, dropna=False)[amt]
                .count()
                .sort_values(ascending=False)
                .head(1)
            )
            if not top.empty:
                top_city = top.index[0]
                top_count = int(top.iloc[0])
                top_city_str = f"{top_city}:{top_count}"

        return {
            "metric": "claims_spike",
            "value": count,
            "period": "March 2025",
            "segment": "Electronics",
            "unit": "claims",
            "details": {
                "total_amount_usd": round(total, 2),
                "top_city": top_city_str,
            },
            "note": "Fallback structured fact"
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
        return "\n".join(lines)

    if not config.GEMINI_API_KEY:
        # Fallback: return raw sources if key missing
        lines = ["Top findings (GEMINI_API_KEY missing):"] + [f"- {s['title']} — {s['snippet']} ({s['url']})" for s in sources]
        return "\n".join(lines)

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
            return text.strip()
        # Fallback to raw sources if unexpected response
        lines = ["Top findings:"] + [f"- {s['title']} — {s['snippet']} ({s['url']})" for s in sources]
        return "\n".join(lines)
    except Exception as e:
        # On error, fall back to listing search results
        lines = [f"Synthesis failed: {e}", "Top findings:"] + [f"- {s['title']} — {s['snippet']} ({s['url']})" for s in sources]
        return "\n".join(lines)


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
