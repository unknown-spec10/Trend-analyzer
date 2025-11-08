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

from .prompts import (
    ROOT_CAUSE_ANALYST_PROMPT,
    get_search_query_prompt,
    get_context_summary_prompt,
    get_relevance_score_prompt,
    get_question_expansion_prompt,
    get_search_decision_prompt
)
from . import config


class AgentState(TypedDict, total=False):
    messages: List[Any]
    internal_fact: Optional[Any]  # structured dict or string
    internal_context: Optional[str]
    internal_summary: Optional[str]  # plain English summary from data
    external_context: Optional[str]
    sources: Optional[List[Dict[str, str]]]  # List of {title, url, snippet}
    final_answer: Optional[str]
    external_relevance: Optional[float]
    needs_external_search: Optional[bool]  # decision flag
    search_iterations: Optional[int]  # track search attempts
    conversation_history: Optional[List[Dict[str, Any]]]  # Previous Q&A pairs
    previous_facts: Optional[List[Dict[str, Any]]]  # Previous data findings
    needs_clarification: Optional[bool]  # Whether question is ambiguous


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
        prompt = get_search_query_prompt(
            metric=metric,
            segment=segment,
            period=period,
            summary=summary[:200]
        )
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
        prompt = get_context_summary_prompt(text=text)
        try:
            resp = summer.invoke(prompt)
            return getattr(resp, "content", str(resp)).strip()
        except Exception:
            return text

    def _judge_relevance(internal_struct: Dict[str, Any], context: str) -> float:
        """LLM-based relevance score between 0.0 and 1.0."""
        judge = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
        brief = json.dumps({k: internal_struct.get(k) for k in ("metric", "value", "period", "segment", "unit")})
        prompt = get_relevance_score_prompt(
            internal_fact_brief=brief,
            context=context
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

    def detect_clarification_need(state: AgentState) -> AgentState:
        """Detect if the question references previous conversation context, and validate question quality."""
        msgs = state.get("messages", [])
        question: str = ""
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("type") == "human":
                q = m.get("content")
                if isinstance(q, str):
                    question = q
                break
            if hasattr(m, "type") and getattr(m, "type") == "human":
                q = getattr(m, "content", None)
                if isinstance(q, str):
                    question = q
                break
            if isinstance(m, HumanMessage):
                question = str(m.content)
                break
        
        # Validate question quality
        import re
        question_stripped = question.strip()
        question_lower = question_stripped.lower()
        
        # Check for invalid/meaningless input
        is_valid = True
        validation_message = None
        
        if not question_stripped:
            is_valid = False
            validation_message = "Please enter a question."
        elif len(question_stripped) < 3:
            is_valid = False
            validation_message = "Your question is too short. Please provide more detail."
        elif re.match(r'^[^a-zA-Z0-9\s]+$', question_stripped):
            # Only special characters/punctuation
            is_valid = False
            validation_message = "Your input appears to be random characters. Please ask a clear question about your data."
        elif not re.search(r'[a-zA-Z]', question_stripped):
            # No alphabetic characters at all
            is_valid = False
            validation_message = "Please use words to ask your question."
        elif len(re.findall(r'\b[a-zA-Z]{2,}\b', question_stripped)) == 0:
            # No valid words (at least 2 letters)
            is_valid = False
            validation_message = "Your input doesn't contain recognizable words. Please ask a clear question."
        else:
            # Check for off-topic/conversational questions (not about data)
            off_topic_patterns = [
                # Greetings and pleasantries
                r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening)',
                r'^(thank you|thanks|thx)',
                # Personal questions
                r'(what is your|what\'s your|tell me your|who are you)',
                r'(your name|your age|where are you|who made you|who created you)',
                # General conversation
                r'^(how are you|how do you do)',
                r'(tell me about yourself|introduce yourself)',
                r'(can you help|help me)',
                # Meta questions about the tool itself
                r'(what can you do|what are you|what is this)',
                r'(how does this work|how do i use)',
            ]
            
            for pattern in off_topic_patterns:
                if re.search(pattern, question_lower):
                    is_valid = False
                    validation_message = "Please ask a question about your data, not a conversational question."
                    break
            
            if is_valid:
                # Check for keyboard mashing (e.g., "lklk;l", "asdfghjkl")
                # Detect: high ratio of consonants, repeated patterns, no vowel-consonant alternation
                words = re.findall(r'\b[a-zA-Z]{3,}\b', question_stripped.lower())
                if words:
                    # Check if it looks like random typing
                    has_meaningful_word = False
                    common_words = {'what', 'which', 'who', 'where', 'when', 'why', 'how', 'show', 'get', 'find', 
                                   'the', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did',
                                   'can', 'could', 'would', 'should', 'may', 'might', 'must', 'will', 'shall',
                                   'top', 'bottom', 'highest', 'lowest', 'most', 'least', 'total', 'sum', 'average',
                                   'region', 'city', 'state', 'country', 'year', 'month', 'day', 'data', 'value'}
                    
                    for word in words:
                        # Check if it's a common word
                        if word in common_words:
                            has_meaningful_word = True
                            break
                        
                        # Check for reasonable vowel-consonant ratio (20-60% vowels is typical)
                        vowels = len([c for c in word if c in 'aeiou'])
                        consonants = len([c for c in word if c.isalpha() and c not in 'aeiou'])
                        if consonants > 0:
                            vowel_ratio = vowels / (vowels + consonants)
                            if 0.2 <= vowel_ratio <= 0.6 and vowels >= 1:
                                has_meaningful_word = True
                                break
                    
                    if not has_meaningful_word and len(words) <= 2:
                        is_valid = False
                        validation_message = "Your input appears to be random typing. Please ask a clear question about your data."
        
        # If invalid, return error state
        if not is_valid:
            logger.warning(f"Invalid question detected: '{question_stripped}'")
            return {
                **state,
                "needs_clarification": True,
                "final_answer": f"❌ Invalid Question\n\n{validation_message}\n\nExamples of valid questions:\n• Which region has the highest claims?\n• What is the average value by category?\n• Show me the top 5 records\n• What trends do you see in the data?",
                "messages": state.get("messages", []) + [
                    {"type": "ai", "content": validation_message}
                ],
            }
        
        conversation_history = state.get("conversation_history", [])
        
        # Check if question has context references
        reference_patterns = [
            r'\b(that|this|it|those|these)\b',
            r'\b(what about|how about|and for)\b',
            r'\b(same|similar|different)\b',
            r'^(females?|males?|women|men)\??$',  # Single word follow-ups
            r'^(other|another)\b',
        ]
        
        import re
        needs_context = False
        if conversation_history:  # Only check if there's history
            for pattern in reference_patterns:
                if re.search(pattern, question.lower()):
                    needs_context = True
                    break
        
        # If needs context, expand question using previous context
        expanded_question = question
        if needs_context and conversation_history:
            prev_qa = conversation_history[-1]  # Most recent Q&A
            prev_question = prev_qa.get("question", "")
            prev_answer = prev_qa.get("answer", "")
            
            # Use LLM to expand the question
            expander = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
            prompt = get_question_expansion_prompt(
                prev_question=prev_question,
                prev_answer=prev_answer[:500],
                current_question=question
            )
            try:
                resp = expander.invoke(prompt)
                expanded = getattr(resp, "content", str(resp)).strip()
                if expanded and len(expanded) > len(question):
                    expanded_question = expanded
                    logger.info(f"Expanded question: '{question}' → '{expanded_question}'")
            except Exception as e:
                logger.warning(f"Question expansion failed: {e}")
        
        return {
            **state,
            "needs_clarification": needs_context,
            "messages": state.get("messages", []) + [
                {"type": "system", "content": f"Resolved question: {expanded_question}"}
            ] if needs_context else state.get("messages", []),
        }

    def call_data_tool(state: AgentState) -> AgentState:
        # Expect the last human message to contain the question
        msgs = state.get("messages", [])
        question = None
        
        # Check if we have an expanded question from clarification
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("type") == "system" and "Resolved question:" in str(m.get("content", "")):
                resolved = m.get("content", "").replace("Resolved question:", "").strip()
                if resolved:
                    question = resolved
                    break
        
        # Otherwise get the original question
        if not question:
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

        # Validate that we got meaningful data from the dataset
        # Check if the result is empty, trivial, or indicates the question doesn't match the data
        is_valid_result = True
        error_message = None

        # Pre-compute value/details status
        value_is_empty = (
            value is None or (isinstance(value, str) and value.lower() in ['none', 'null', 'unknown', 'n/a', ''])
        )
        details_obj = struct.get("details") if isinstance(struct, dict) else None
        details_has_content = False
        if isinstance(details_obj, dict):
            for v in details_obj.values():
                if isinstance(v, dict) and len(v) > 0:
                    details_has_content = True
                    break
                if isinstance(v, list) and len(v) > 0:
                    details_has_content = True
                    break

        # Check 1: Explicit error metric (LLM detected mismatch)
        if metric == "error":
            is_valid_result = False
            error_message = "Question doesn't match dataset columns"

        # Check 2: No meaningful value extracted AND no useful details
        elif value_is_empty and not details_has_content:
            is_valid_result = False
            error_message = "Unable to find relevant data"

        # Check 3: Metric is unknown (indicates data extraction failed)
        elif metric == "unknown" or metric is None:
            is_valid_result = False
            error_message = "Unable to understand question in context of dataset"

        # Check 4: Empty or trivial summary
        elif not internal_summary or len(internal_summary.strip()) < 10:
            is_valid_result = False
            error_message = "No meaningful information extracted from dataset"

        # Check 5: Summary indicates error or inability to answer
        elif any(phrase in internal_summary.lower() for phrase in [
            'unable to', 'cannot find', 'no data', 'does not exist', 'not found',
            'not available', 'no column', 'no information', 'not present in'
        ]):
            is_valid_result = False
            error_message = "Question doesn't match available data columns"
        
        # If validation failed, return error state
        if not is_valid_result:
            logger.warning(f"Invalid data extraction for question. Reason: {error_message}")
            
            # Get the original human question for error message
            original_question = question
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("type") == "human":
                    original_question = m.get("content")
                    break
                if isinstance(m, HumanMessage):
                    original_question = m.content
                    break
            
            return {
                "messages": msgs + [{"type": "ai", "content": f"Data validation failed: {error_message}"}],
                "internal_fact": struct,
                "internal_context": None,
                "internal_summary": None,
                "final_answer": f"""❌ Invalid Question for This Dataset

{error_message}. Your question might be asking about:
• Data columns that don't exist in this dataset
• Concepts or metrics not present in the data
• Information that requires external knowledge

**Suggestions:**
1. Check what columns are available in your dataset
2. Rephrase your question using column names from the data
3. Click on suggested questions in the sidebar for examples

**Example Valid Questions:**
• What is the total/average of [numeric column]?
• Which [category] has the highest [value]?
• Show me the distribution of [column]
• Compare [column1] across different [column2]""",
                "needs_clarification": True,
            }

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
        question = str(question or "")
        
        internal_summary = state.get("internal_summary") or ""
        internal_struct = _coerce_structured_fact(state.get("internal_fact"))
        
        # Use LLM to judge if internal data is sufficient
        judge = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
        prompt = get_search_decision_prompt(
            question=question,
            internal_summary=internal_summary,
            internal_struct=internal_struct
        )
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
        best_sources = []
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

            # Parse JSON response to extract summary and sources
            ext_text = str(summary)
            sources_list = []
            try:
                parsed = json.loads(str(summary))
                if isinstance(parsed, dict):
                    ext_text = parsed.get("summary", str(summary))
                    sources_list = parsed.get("sources", [])
            except Exception:
                # If not JSON, use as-is
                pass

            # Auto-summarize the external context
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
                best_score, best_text, best_sources = score, ext_text, sources_list
            if score >= threshold:
                logger.info(f"Relevance threshold reached ({score:.2f} >= {threshold})")
                break

        return {
            "messages": msgs + [{"type": "ai", "content": f"External context gathered (relevance {best_score:.2f}): {best_text}"}],
            "external_context": best_text,
            "sources": best_sources,
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
        
        # Update previous_facts list
        previous_facts = state.get("previous_facts", []) or []
        previous_facts.append(internal_struct)
        
        return {
            "messages": msgs + [{"type": "ai", "content": final_text}],
            "final_answer": final_text,
            "previous_facts": previous_facts,
        }

    # Conditional routing functions
    def route_after_clarification(state: AgentState) -> str:
        """Route to END if validation failed, otherwise continue to data tool."""
        # If final_answer is already set by validation, skip to END
        if state.get("final_answer") and state.get("needs_clarification"):
            return "END"
        return "call_data_tool"
    
    def route_after_data_tool(state: AgentState) -> str:
        """Route to END if data extraction failed, otherwise continue to decision."""
        # If final_answer is set after data_tool (validation failed), skip to END
        if state.get("final_answer") and state.get("needs_clarification"):
            return "END"
        return "decide_if_search_needed"
    
    def route_after_decision(state: AgentState) -> str:
        """Route to search or directly to synthesis based on decision."""
        if state.get("needs_external_search", True):
            return "call_search_tool"
        return "synthesize_answer"

    # Build the graph with intelligent routing and clarification detection
    graph = StateGraph(AgentState)
    graph.add_node("detect_clarification_need", detect_clarification_need)
    graph.add_node("call_data_tool", call_data_tool)
    graph.add_node("decide_if_search_needed", decide_if_search_needed)
    graph.add_node("call_search_tool", call_search_tool)
    graph.add_node("synthesize_answer", synthesize_answer)

    graph.set_entry_point("detect_clarification_need")
    graph.add_conditional_edges(
        "detect_clarification_need",
        route_after_clarification,
        {
            "call_data_tool": "call_data_tool",
            "END": END,
        }
    )
    graph.add_conditional_edges(
        "call_data_tool",
        route_after_data_tool,
        {
            "decide_if_search_needed": "decide_if_search_needed",
            "END": END,
        }
    )
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
