"""Agent orchestration using LangGraph (SRP).

Intelligent adaptive workflow:
START -> call_data_tool -> decide_if_search_needed -> [conditional routing]
  -> If search needed: call_search_tool -> synthesize_answer -> END
  -> If data sufficient: synthesize_answer -> END

The agent dynamically adapts to ANY dataset and only searches the web when
the internal data is insufficient to answer the user's question.
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
    internal_summary: Optional[str]  # plain English summary from data
    external_context: Optional[str]
    final_answer: Optional[str]
    external_relevance: Optional[float]
    needs_external_search: Optional[bool]  # decision flag
    search_iterations: Optional[int]  # track search attempts


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

    def _build_search_payload(struct: Dict[str, Any], attempt: int, summary: str = "") -> Dict[str, Any]:
        """Build adaptive Google CSE query based on data context."""
        segment = str(struct.get("segment") or "").strip()
        period = str(struct.get("period") or "").strip()
        metric = str(struct.get("metric") or "").strip()
        
        # Use LLM to generate contextual search query
        query_builder = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
        prompt = f"""Generate a focused web search query to find external factors explaining this data pattern.

DATA PATTERN:
- Metric: {metric}
- Value/Segment: {segment}
- Period: {period}
- Summary: {summary[:200]}

Generate a search query that:
1. Focuses on CAUSES, TRENDS, or EXTERNAL FACTORS
2. Uses specific terms from the data (region, demographic, category)
3. Avoids generic terms
4. Is 5-10 words long

Respond with ONLY the search query, no explanation.
"""
        try:
            resp = query_builder.invoke(prompt)
            query = getattr(resp, "content", str(resp)).strip()
            # Fallback if LLM fails
            if not query or len(query) < 10:
                query = f"{segment} {metric} {period} causes trends analysis"
        except Exception:
            query = f"{segment} {metric} {period} causes trends analysis"

        windows = ["m6", "y1", "d30"]
        date_restrict = windows[min(attempt, len(windows) - 1)]

        # Minimal negative keywords - let search be broad
        negative_keywords = ["recipe", "entertainment", "celebrity"]

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
        
        # Extract the plain-English summary from struct if available
        internal_summary = struct.get("summary", "")

        return {
            "messages": msgs + [
                {"type": "ai", "content": f"Internal fact found: {struct}"},
                {"type": "ai", "content": f"Internal context summary:\n{internal_context}"} if internal_context else {"type": "ai", "content": ""}
            ],
            "internal_fact": struct,
            "internal_context": internal_context or None,
            "internal_summary": internal_summary or None,
        }

    def decide_if_search_needed(state: AgentState) -> AgentState:
        """Intelligent decision node: determines if external search is necessary."""
        question = None
        for m in state.get("messages", [])[::-1]:
            if isinstance(m, dict) and m.get("type") == "human":
                question = m.get("content")
                break
            if isinstance(m, HumanMessage):
                question = m.content
                break
        question = question or ""
        
        internal_summary = state.get("internal_summary") or ""
        internal_struct = _coerce_structured_fact(state.get("internal_fact"))
        
        # Use LLM to judge if internal data is sufficient
        judge = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
        prompt = f"""You are analyzing whether a data-driven answer is sufficient or needs external research.

USER QUESTION:
{question}

INTERNAL DATA ANSWER:
{internal_summary}

INTERNAL DATA DETAILS:
{json.dumps(internal_struct, indent=2)}

DECISION TASK:
Determine if the internal data provides a COMPLETE answer to the user's question.

Answer "YES" if:
- The question asks about patterns, counts, distributions, or comparisons IN THE DATA
- The data directly answers what/who/where/how many/how much
- The question is purely descriptive or analytical

Answer "NO" if:
- The question asks WHY something happened (needs causal explanation)
- The question asks about external factors, industry trends, or real-world events
- The question asks about causes, reasons, or drivers beyond the data
- The data shows a pattern but doesn't explain the underlying cause

Respond with ONLY "YES" or "NO" followed by a brief one-line reason.
Format: YES|reason or NO|reason
"""
        try:
            resp = judge.invoke(prompt)
            decision_text = getattr(resp, "content", str(resp)).strip().upper()
            
            # Parse decision
            needs_search = True  # default to searching
            if decision_text.startswith("YES"):
                needs_search = False
                logger.info("Decision: Internal data is sufficient. Skipping external search.")
            else:
                logger.info("Decision: External search needed to supplement internal data.")
                
        except Exception as e:
            logger.warning(f"Decision node failed: {e}. Defaulting to search.")
            needs_search = True
        
        return {
            **state,
            "needs_external_search": needs_search,
            "search_iterations": 0,
        }

    def call_search_tool(state: AgentState) -> AgentState:
        """Perform adaptive web search with relevance scoring and iteration."""
        fact_any = state.get("internal_fact") or {}
        struct = _coerce_structured_fact(fact_any)
        internal_summary = state.get("internal_summary") or ""
        msgs = state.get("messages", [])
        current_iteration = state.get("search_iterations", 0)

        best_text = ""
        best_score = -1.0
        threshold = 0.6
        max_attempts = 3
        
        for i in range(max_attempts):
            payload = _build_search_payload(struct, i, internal_summary)
            logger.info(f"Search attempt {i+1}: query='{payload.get('query')}'")
            
            try:
                t0 = time.perf_counter()
                summary = search_tool.invoke(json.dumps(payload))
                logger.info("search attempt=%d duration_ms=%.1f", i + 1, (time.perf_counter() - t0) * 1000.0)
            except Exception as e:
                summary = f"Search failed: {e}"
                logger.warning(f"Search attempt {i+1} failed: {e}")

            # Auto-summarize the external context
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

            # Enforce tight bullet-first format
            ext_text = _tight_bulleted_summary(ext_text)

            score = _judge_relevance(struct, ext_text)
            logger.info(f"Search attempt {i+1} relevance score: {score:.2f}")

            if score > best_score:
                best_score, best_text = score, ext_text
            if score >= threshold:
                logger.info(f"Relevance threshold reached ({score:.2f} >= {threshold})")
                break

        return {
            "messages": msgs + [{"type": "ai", "content": f"External context gathered (relevance {best_score:.2f}): {best_text}"}],
            "external_context": best_text,
            "external_relevance": max(0.0, best_score),
            "search_iterations": (current_iteration or 0) + 1,
        }

    def synthesize_answer(state: AgentState) -> AgentState:
        """Create final answer, adapting format based on whether external search was used."""
        question = None
        for m in state.get("messages", [])[::-1]:
            if isinstance(m, dict) and m.get("type") == "human":
                question = m.get("content")
                break
            if isinstance(m, HumanMessage):
                question = m.content
                break
        question = question or "What does the data show?"

        internal_struct = _coerce_structured_fact(state.get("internal_fact"))
        internal_summary = state.get("internal_summary") or ""
        internal_for_prompt = json.dumps(internal_struct, ensure_ascii=False)
        external = state.get("external_context") or ""
        used_search = state.get("needs_external_search", False)

        system = ROOT_CAUSE_ANALYST_PROMPT
        
        if not used_search or not external:
            # Data-only answer: concise, factual, no external citations needed
            user = (
                f"Question: {question}\n\n"
                f"Internal Data Analysis:\n{internal_summary}\n\n"
                f"Structured Details: {internal_for_prompt}\n\n"
                "Write a concise, data-driven answer:\n"
                "- 3-5 bullets highlighting key findings from the data\n"
                "- 1-2 sentences synthesizing the pattern or insight\n"
                "- Note that this answer is based entirely on the provided dataset\n"
            )
        else:
            # Combined answer: data + external research
            user = (
                f"Question: {question}\n\n"
                f"Internal Data: {internal_summary}\n"
                f"Details: {internal_for_prompt}\n\n"
                f"External Research:\n{external}\n\n"
                "Write a comprehensive answer blending data and research:\n"
                "- 5-7 concise bullets citing sources inline as (Title - URL)\n"
                "- A short synthesis paragraph (3-4 sentences) connecting data to external factors\n"
                "- 'Confidence' line explaining how well data + research answer the question\n"
                "- 'Limitations' line noting any data gaps or assumptions\n"
            )
        
        t0 = time.perf_counter()
        response = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
        logger.info("synthesis duration_ms=%.1f", (time.perf_counter() - t0) * 1000.0)
        final_text = getattr(response, "content", str(response))

        msgs = state.get("messages", [])
        return {"messages": msgs + [{"type": "ai", "content": final_text}], "final_answer": final_text}

    # Conditional routing function
    def route_after_decision(state: AgentState) -> str:
        """Route to search or directly to synthesis based on decision."""
        if state.get("needs_external_search", True):
            return "call_search_tool"
        return "synthesize_answer"

    # Build the graph with intelligent routing
    graph = StateGraph(AgentState)
    graph.add_node("call_data_tool", call_data_tool)
    graph.add_node("decide_if_search_needed", decide_if_search_needed)
    graph.add_node("call_search_tool", call_search_tool)
    graph.add_node("synthesize_answer", synthesize_answer)

    graph.set_entry_point("call_data_tool")
    graph.add_edge("call_data_tool", "decide_if_search_needed")
    graph.add_conditional_edges(
        "decide_if_search_needed",
        route_after_decision,
        {
            "call_search_tool": "call_search_tool",
            "synthesize_answer": "synthesize_answer",
        }
    )
    graph.add_edge("call_search_tool", "synthesize_answer")
    graph.add_edge("synthesize_answer", END)

    return graph.compile()
