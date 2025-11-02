"""Agent orchestration using LangGraph (SRP).

Defines a strict sequence:
START -> call_data_tool -> call_search_tool -> synthesize_answer -> END
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
import json
import logging
import time

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from .prompts import ROOT_CAUSE_ANALYST_PROMPT
from . import config


class AgentState(TypedDict, total=False):
    messages: List[Any]
    internal_fact: Optional[Any]  # structured dict or string
    internal_context: Optional[str]
    external_context: Optional[str]
    final_answer: Optional[str]
    external_relevance: Optional[float]


def create_agent_executor(tools: List[Any]):
    """Create and compile the LangGraph agent with provided tools.

    tools: a list of LangChain Tool objects. Must include names:
      - "query_data"
      - "search_for_probable_causes"
    """
    # Resolve tools by name from provided list
    tool_map = {getattr(t, "name", None): t for t in tools}
    data_tool = tool_map.get("query_data")
    search_tool = tool_map.get("search_for_probable_causes")
    summarize_tool = tool_map.get("summarize_text")

    if data_tool is None or search_tool is None:
        missing = [n for n in ("query_data", "search_for_probable_causes") if tool_map.get(n) is None]
        raise ValueError(f"Missing required tools: {', '.join(missing)}")

    # Default Groq model for synthesis (GROQ_API_KEY is read from the environment)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3
    )
    logger = logging.getLogger(__name__)

    def _coerce_structured_fact(fact: Any) -> Dict[str, Any]:
        """Coerce tool output into a structured dict if possible."""
        if isinstance(fact, dict):
            return fact
        if isinstance(fact, str):
            try:
                parsed = json.loads(fact)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return {"metric": "unknown", "value": fact, "period": "unknown", "segment": "unknown", "unit": "unknown"}
        return {"metric": "unknown", "value": str(fact), "period": "unknown", "segment": "unknown", "unit": "unknown"}

    def _build_search_payload(struct: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Build a focused Google CSE query with dateRestrict and negative keywords."""
        segment = str(struct.get("segment") or "").strip()
        period = str(struct.get("period") or "recent period").strip()
        base_terms = [segment, "claims spike", period, "causes", "industry news", "root cause", "recall", "defect", "outage", "power surge", "supply chain", "weather", "storm", "warranty"]
        query = " ".join(t for t in base_terms if t)

        windows = ["m6", "y1", "d30"]
        date_restrict = windows[min(attempt, len(windows) - 1)]

        negative_keywords = [
            "medical", "health", "dialysis", "inflammation", "cardiovascular", "patient", "clinical",
        ]

        return {
            "query": query,
            "dateRestrict": date_restrict,
            "negative_keywords": negative_keywords,
        }

    def _tight_bulleted_summary(text: str) -> str:
        """Create a bullet-first, length-constrained summary of external context."""
        summer = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
        prompt = (
            "Summarize the CONTEXT into tight bullets first, then one short wrap-up line.\n"
            "Rules:\n"
            "- 5-7 bullets, each <= 18 words.\n"
            "- Max 1200 characters total.\n"
            "- Preserve any inline citations like (Title - URL) if present.\n"
            "- Focus on causal drivers and evidence.\n\n"
            f"CONTEXT:\n{text}"
        )
        try:
            resp = summer.invoke(prompt)
            return getattr(resp, "content", str(resp)).strip()
        except Exception:
            return text

    def _judge_relevance(internal_struct: Dict[str, Any], context: str) -> float:
        """LLM-based relevance score between 0.0 and 1.0."""
        judge = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
        brief = json.dumps({k: internal_struct.get(k) for k in ("metric", "value", "period", "segment", "unit")})
        prompt = (
            "Score how relevant the CONTEXT is to the INTERNAL FACT on a 0.0-1.0 scale.\n"
            "Return ONLY a number with up to 2 decimals, no text.\n\n"
            f"INTERNAL FACT: {brief}\n\nCONTEXT:\n{context}\n\nScore:"
        )
        try:
            resp = judge.invoke(prompt)
            txt = getattr(resp, "content", str(resp)).strip()
            import re as _re
            m = _re.search(r"\d+(?:\.\d+)?", txt)
            if m:
                return max(0.0, min(1.0, float(m.group(0))))
        except Exception:
            pass
        return 0.0

    def call_data_tool(state: AgentState) -> AgentState:
        # Expect the last human message to contain the question
        msgs = state.get("messages", [])
        question = None
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("type") == "human":
                question = m.get("content")
                break
            if hasattr(m, "type") and getattr(m, "type") == "human":
                question = getattr(m, "content", None)
                break
            if isinstance(m, HumanMessage):
                question = m.content
                break
        question = question or "What is the key internal fact?"

        # Invoke the tool with the user's question
        t0 = time.perf_counter()
        fact = data_tool.invoke(question)
        logger.info("data_tool duration_ms=%.1f", (time.perf_counter() - t0) * 1000.0)
        struct = _coerce_structured_fact(fact)

        # Build a concise internal context summary from the structured fact
        parts: list[str] = []
        metric = struct.get("metric")
        value = struct.get("value")
        unit = struct.get("unit")
        period = struct.get("period")
        segment = struct.get("segment")
        if metric is not None:
            parts.append(f"• Metric: {metric}")
        if value is not None:
            if unit:
                parts.append(f"• Value: {value} {unit}")
            else:
                parts.append(f"• Value: {value}")
        if period:
            parts.append(f"• Period: {period}")
        if segment:
            parts.append(f"• Segment: {segment}")
        details = struct.get("details")
        if isinstance(details, dict):
            # Surface common details fields if present
            if details.get("total_amount_usd") is not None:
                parts.append(f"• Total amount: ${details['total_amount_usd']:,}")
            if details.get("top_city"):
                parts.append(f"• Top city: {details['top_city']}")
        internal_context = "\n".join(parts) if parts else ""

        return {
            "messages": msgs + [
                {"type": "ai", "content": f"Internal fact found: {struct}"},
                {"type": "ai", "content": f"Internal context summary:\n{internal_context}"} if internal_context else {"type": "ai", "content": ""}
            ],
            "internal_fact": struct,
            "internal_context": internal_context or None,
        }

    def call_search_tool(state: AgentState) -> AgentState:
        fact_any = state.get("internal_fact") or {}
        struct = _coerce_structured_fact(fact_any)
        msgs = state.get("messages", [])

        best_text = ""
        best_score = -1.0
        threshold = 0.6
        attempts = 3
        for i in range(attempts):
            payload = _build_search_payload(struct, i)
            try:
                t0 = time.perf_counter()
                summary = search_tool.invoke(json.dumps(payload))
                logger.info("search attempt=%d duration_ms=%.1f", i + 1, (time.perf_counter() - t0) * 1000.0)
            except Exception as e:
                summary = f"Search failed: {e}"

            # Auto-summarize the external context to reduce token usage and tighten synthesis
            ext_text = str(summary)
            if summarize_tool is not None and isinstance(ext_text, str) and ext_text.strip():
                try:
                    t1 = time.perf_counter()
                    short = summarize_tool.invoke(ext_text)
                    logger.info("auto_summarize duration_ms=%.1f", (time.perf_counter() - t1) * 1000.0)
                    if isinstance(short, str) and short.strip():
                        ext_text = short
                except Exception:
                    pass

            # Enforce tight bullet-first format and length budget
            ext_text = _tight_bulleted_summary(ext_text)

            score = _judge_relevance(struct, ext_text)

            if score > best_score:
                best_score, best_text = score, ext_text
            if score >= threshold:
                break

        return {
            "messages": msgs + [{"type": "ai", "content": f"External context gathered (relevance {best_score:.2f}): {best_text}"}],
            "external_context": best_text,
            "external_relevance": max(0.0, best_score),
        }

    def synthesize_answer(state: AgentState) -> AgentState:
        question = None
        for m in state.get("messages", [])[::-1]:
            if isinstance(m, dict) and m.get("type") == "human":
                question = m.get("content")
                break
            if isinstance(m, HumanMessage):
                question = m.content
                break
        question = question or "Why did this happen?"

        internal_struct = _coerce_structured_fact(state.get("internal_fact"))
        internal_for_prompt = json.dumps(internal_struct, ensure_ascii=False)
        external = state.get("external_context") or ""

        system = ROOT_CAUSE_ANALYST_PROMPT
        user = (
            f"Question: {question}\n\n"
            f"Internal Fact (structured JSON): {internal_for_prompt}\n\n"
            f"External Context (summarized, bulleted):\n{external}\n\n"
            "Write the final response using this strict format:\n"
            "- Start with 5-7 concise bullets (<= 18 words each) citing sources inline as (Title - URL).\n"
            "- Then a short paragraph (<= 4 sentences) synthesizing causal drivers.\n"
            "- Add a 'Confidence/Relevance' line with a brief rationale tied to the cited sources.\n"
            "- Add an 'Assumptions/Unknowns' line listing 1-3 key uncertainties.\n"
        )
        t0 = time.perf_counter()
        response = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
        logger.info("synthesis duration_ms=%.1f", (time.perf_counter() - t0) * 1000.0)
        final_text = getattr(response, "content", str(response))

        # Optionally summarize the final answer for a crisp executive insight
        summarized: str | None = None
        if summarize_tool is not None and isinstance(final_text, str) and final_text.strip():
            try:
                summarized = summarize_tool.invoke(final_text)
            except Exception:
                summarized = None

        # Prefer summary-only final output; fall back to full answer if summarization unavailable
        final_output = (
            summarized.strip() if isinstance(summarized, str) and summarized.strip() else final_text
        )
        msgs = state.get("messages", [])
        return {"messages": msgs + [{"type": "ai", "content": final_output}], "final_answer": final_output}

    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("call_data_tool", call_data_tool)
    graph.add_node("call_search_tool", call_search_tool)
    graph.add_node("synthesize_answer", synthesize_answer)

    graph.set_entry_point("call_data_tool")
    graph.add_edge("call_data_tool", "call_search_tool")
    graph.add_edge("call_search_tool", "synthesize_answer")
    graph.add_edge("synthesize_answer", END)

    return graph.compile()
