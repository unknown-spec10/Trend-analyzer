"""Agent orchestration using LangGraph (SRP).

Intelligent adaptive workflow:
START -> call_data_tool -> decide_if_search_needed -> [conditional routing]
  -> If search needed: call_search_tool -> synthesize_answer -> END
  -> If data sufficient: synthesize_answer -> END

The agent dynamically adapts to ANY dataset and only searches the web when
the internal data is insufficient to answer the user's question.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, cast
import json
import logging
import time

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from .groq_wrapper import GroqWithGeminiFallback

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
    llm = GroqWithGeminiFallback(
        model="llama-3.1-8b-instant",
        temperature=0.2
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
        
        # Use LLM to generate contextual search query (with Gemini fallback)
        prompt = get_search_query_prompt(
            metric=metric,
            segment=segment,
            period=period,
            summary=summary[:200]
        )
        
        query = ""
        try:
            query_builder = GroqWithGeminiFallback(model="llama-3.1-8b-instant", temperature=0.3)
            resp = query_builder.invoke(prompt)
            query = getattr(resp, "content", str(resp)).strip()
        except Exception as e:
            error_str = str(e).lower()
            if "429" in str(e) or "rate limit" in error_str or "quota" in error_str:
                logger.warning("Groq rate limit in search query builder. Using Gemini fallback.")
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=config.GEMINI_API_KEY)
                    model = genai.GenerativeModel("gemini-2.0-flash-exp")
                    
                    prompt_text = "\n".join([str(msg.get("content", msg)) if isinstance(msg, dict) else str(msg) for msg in (prompt if isinstance(prompt, list) else [prompt])])
                    response = model.generate_content(prompt_text)
                    query = response.text.strip()
                except Exception as fallback_error:
                    logger.error(f"Gemini fallback failed: {fallback_error}")
                    query = ""
            else:
                logger.warning(f"Query builder failed: {e}")
                query = ""
        
        # Fallback if LLM fails or returns empty
        if not query or len(query) < 10:
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
        summer = GroqWithGeminiFallback(model="llama-3.1-8b-instant", temperature=0.2)
        prompt = get_context_summary_prompt(text=text)
        try:
            resp = summer.invoke(prompt)
            return getattr(resp, "content", str(resp)).strip()
        except Exception as e:
            error_str = str(e).lower()
            if "429" in str(e) or "rate limit" in error_str or "quota" in error_str:
                logger.warning("Groq rate limit in context summarizer. Using Gemini fallback.")
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=config.GEMINI_API_KEY)
                    model = genai.GenerativeModel("gemini-2.0-flash-exp")

                    prompt_text = "\n".join([str(msg.get("content", msg)) if isinstance(msg, dict) else str(msg) for msg in (prompt if isinstance(prompt, list) else [prompt])])
                    response = model.generate_content(prompt_text)
                    return response.text.strip()
                except Exception as fallback_error:
                    logger.error(f"Gemini fallback failed: {fallback_error}")
                    return text
            return text

    def _judge_relevance(internal_struct: Dict[str, Any], context: str) -> float:
        """LLM-based relevance score between 0.0 and 1.0."""
        judge = GroqWithGeminiFallback(model="llama-3.1-8b-instant", temperature=0.0)
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
        except Exception as e:
            error_str = str(e).lower()
            if "429" in str(e) or "rate limit" in error_str or "quota" in error_str:
                logger.warning("Groq rate limit in relevance judge. Using Gemini fallback.")
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=config.GEMINI_API_KEY)
                    model = genai.GenerativeModel("gemini-2.0-flash-exp")

                    prompt_text = "\n".join([str(msg.get("content", msg)) if isinstance(msg, dict) else str(msg) for msg in (prompt if isinstance(prompt, list) else [prompt])])
                    response = model.generate_content(prompt_text)
                    txt = response.text.strip()
                    import re as _re
                    m = _re.search(r"\d+(?:\.\d+)?", txt)
                    if m:
                        return max(0.0, min(1.0, float(m.group(0))))
                except Exception as fallback_error:
                    logger.error(f"Gemini fallback failed: {fallback_error}")
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
                # Use ONLY linguistic patterns - NO hardcoded word lists
                words = re.findall(r'\b[a-zA-Z]{3,}\b', question_stripped.lower())
                if words:
                    # Check if it looks like random typing using purely structural heuristics
                    has_meaningful_word = False
                    
                    for word in words:
                        # Check for reasonable vowel-consonant ratio (20-70% vowels is typical English)
                        # This avoids hardcoding while catching gibberish like "lklklk" or "zxcvbn"
                        vowels = len([c for c in word if c in 'aeiou'])
                        consonants = len([c for c in word if c.isalpha() and c not in 'aeiou'])
                        if consonants > 0:
                            vowel_ratio = vowels / (vowels + consonants)
                            if 0.2 <= vowel_ratio <= 0.7 and vowels >= 1:
                                has_meaningful_word = True
                                break
                        elif vowels >= 1:  # All vowels is still valid (e.g., "a", "I", "area")
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
                "final_answer": f"❌ Invalid Question\n\n{validation_message}\n\nExamples of valid questions:\n• What is the average value by category?\n• Which item has the highest total?\n• Show me the top 5 records\n• What trends do you see in the data?",
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
            expander = GroqWithGeminiFallback(model="llama-3.1-8b-instant", temperature=0.0)
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
        
        # Clean up messy formatting in summary (remove extra spaces, fix number formatting)
        if internal_summary:
            import re
            # Fix numbers that got split with spaces (e.g., "1 , 121.87" -> "1,121.87")
            internal_summary = re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', internal_summary)
            # Fix currency that got separated (e.g., "$ 12" -> "$12")
            internal_summary = re.sub(r'\$\s+', r'$', internal_summary)
            # Fix ranges that got mangled (e.g., "87to63" -> "87 to 63")
            internal_summary = re.sub(r'(\d)([a-z]+)(\d)', r'\1 \2 \3', internal_summary)
            # Remove excessive spaces
            internal_summary = re.sub(r'\s+', ' ', internal_summary)
            # Fix parentheses spacing (e.g., "( 123" -> "(123")
            internal_summary = re.sub(r'\(\s+', '(', internal_summary)
            internal_summary = re.sub(r'\s+\)', ')', internal_summary)
            internal_summary = internal_summary.strip()

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
        
        # Use LLM to judge if internal data is sufficient (with Gemini fallback)
        prompt = get_search_decision_prompt(
            question=question,
            internal_summary=internal_summary,
            internal_struct=internal_struct
        )
        
        decision_text = ""
        try:
            # Try Groq first
            judge = GroqWithGeminiFallback(model="llama-3.1-8b-instant", temperature=0.0)
            resp = judge.invoke(prompt)
            decision_text = getattr(resp, "content", str(resp)).strip().upper()
            
        except Exception as e:
            error_str = str(e).lower()
            # Check if it's a rate limit error
            if "429" in str(e) or "rate limit" in error_str or "quota" in error_str:
                logger.warning(f"Groq rate limit in decision node. Falling back to Gemini.")
                try:
                    # Fallback to Gemini
                    import google.generativeai as genai
                    genai.configure(api_key=config.GEMINI_API_KEY)
                    model = genai.GenerativeModel("gemini-2.0-flash-exp")
                    
                    # Convert prompt to string
                    if isinstance(prompt, list):
                        prompt_text = "\n".join([str(msg.get("content", msg)) if isinstance(msg, dict) else str(msg) for msg in prompt])
                    else:
                        prompt_text = str(prompt)
                    
                    response = model.generate_content(prompt_text)
                    decision_text = response.text.strip().upper()
                    logger.info("Successfully used Gemini fallback for decision node")
                except Exception as fallback_error:
                    logger.error(f"Gemini fallback also failed: {fallback_error}. Defaulting to search.")
                    decision_text = "NO"  # Default to search
            else:
                logger.warning(f"Decision node failed: {e}. Defaulting to search.")
                decision_text = "NO"  # Default to search
        
        # Parse decision
        needs_search = True  # default to searching
        if decision_text.startswith("YES"):
            needs_search = False
            logger.info("Decision: Internal data is sufficient. Skipping external search.")
        else:
            logger.info("Decision: External search needed to supplement internal data.")
        
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
            # Data-only answer: Use the clean summary from the data tool directly
            # The summary is already well-formatted with proper numbers and structure
            details = internal_struct.get("details", {})
            
            # Build a clean, formatted answer from the structured data
            answer_parts = []
            answer_parts.append(f"**{question}**\n")
            answer_parts.append(f"\n{internal_summary}\n")
            
            # Add detailed breakdown if available in details
            if isinstance(details, dict):
                # Check for multi-metric details (min/max/mean by category)
                has_multi_metric = any(key.startswith("by_") for key in details.keys())
                if has_multi_metric:
                    answer_parts.append("\n**Detailed Breakdown:**")
                    for key, value in details.items():
                        if key.startswith("by_") and isinstance(value, dict):
                            category_name = key.replace("by_", "").replace("_", " ").title()
                            answer_parts.append(f"\n{category_name}:")
                            for cat, metrics in value.items():
                                if isinstance(metrics, dict):
                                    metrics_str = ", ".join([f"{k}: ${v:,.2f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                                                            for k, v in metrics.items()])
                                    answer_parts.append(f"  • {cat.capitalize()}: {metrics_str}")
                                else:
                                    answer_parts.append(f"  • {cat}: {metrics:,.2f}" if isinstance(metrics, (int, float)) else f"  • {cat}: {metrics}")
            
            final_text = "\n".join(answer_parts)
            
            # Skip LLM synthesis for data-only questions to preserve clean formatting
            msgs = state.get("messages", [])
            previous_facts = state.get("previous_facts", []) or []
            previous_facts.append(internal_struct)
            
            return {
                "messages": msgs + [{"type": "ai", "content": final_text}],
                "final_answer": final_text,
                "previous_facts": previous_facts,
            }
        else:
            # Combined answer: data + external research - use LLM to synthesize
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
            try:
                response = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
                logger.info("synthesis duration_ms=%.1f", (time.perf_counter() - t0) * 1000.0)
                final_text = getattr(response, "content", str(response))
            except Exception as e:
                error_str = str(e).lower()
                logger.warning(f"Synthesis LLM failed: {e}")
                
                # If rate limit or other Groq error, use Gemini fallback
                if "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
                    logger.info("Rate limit detected in synthesis, using Gemini fallback")
                else:
                    logger.info("Groq synthesis failed, attempting Gemini fallback")
                
                try:
                    import google.generativeai as genai
                    from generic_analyst_agent.src.config import GEMINI_API_KEY
                    
                    genai.configure(api_key=GEMINI_API_KEY)
                    gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
                    
                    gemini_prompt = f"{system}\n\n{user}"
                    gemini_response = gemini_model.generate_content(gemini_prompt)
                    final_text = gemini_response.text
                    logger.info("Gemini synthesis successful")
                except Exception as gemini_error:
                    logger.error(f"Gemini fallback also failed: {gemini_error}")
                    # Return a basic formatted answer as last resort
                    final_text = f"**Answer:**\n\n{internal_summary}\n\n(Note: Enhanced synthesis unavailable due to API limitations)"

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

    compiled = graph.compile()

    class AgentExecutor:
        """Wrapper providing both invoke() and stream() execution modes."""

        def __init__(self):
            self._compiled = compiled

        def invoke(self, state: AgentState) -> Any:  # type: ignore[override]
            return self._compiled.invoke(state)

        def stream(self, state: Dict[str, Any]):
            """Yield step-by-step progress events with incremental state updates.

            Each yielded item is a dict like:
            {"stage": str, "message": str | None, "state": AgentState | None, ...}
            """
            # Step 1: Clarification / validation
            s0 = cast(AgentState, state)
            yield {"stage": "start", "message": "Analyzing question...", "state": s0}
            s1 = detect_clarification_need(s0)
            last_msg = None
            if s1.get("messages"):
                msgs_list = s1.get("messages") or []
                if isinstance(msgs_list, list) and len(msgs_list) > 0 and isinstance(msgs_list[-1], dict):
                    last_msg = msgs_list[-1].get("content")
            yield {"stage": "clarify", "message": last_msg, "state": s1}

            if route_after_clarification(s1) == "END":
                yield {"stage": "end", "message": "Stopped after validation.", "state": s1, "final": True}
                return

            # Step 2: Data tool
            yield {"stage": "data", "message": "Querying dataset...", "state": s1}
            s2 = call_data_tool(s1)
            internal_summary = s2.get("internal_summary") or ""
            msg2 = f"Internal summary: {internal_summary}" if internal_summary else "Internal data extracted."
            yield {"stage": "data", "message": msg2, "state": s2}

            if route_after_data_tool(s2) == "END":
                yield {"stage": "end", "message": "Answer produced from validation outcome.", "state": s2, "final": True}
                return

            # Step 3: Decision
            yield {"stage": "decision", "message": "Deciding whether to search the web...", "state": s2}
            s3 = decide_if_search_needed(s2)
            needs = s3.get("needs_external_search", True)
            yield {"stage": "decision", "message": ("External search is needed" if needs else "Internal data is sufficient"), "state": s3}

            # Step 4 (optional): Search
            s4 = s3
            if needs:
                yield {"stage": "search", "message": "Gathering external context...", "state": s3}
                s4 = call_search_tool(s3)
                ext = (s4.get("external_context") or "")[:200]
                yield {"stage": "search", "message": f"External context summarized: {ext}...", "state": s4}

            # Step 5: Synthesis
            yield {"stage": "synthesis", "message": "Composing the final answer...", "state": s4}
            s5 = synthesize_answer(s4)
            yield {"stage": "end", "message": "Done.", "state": s5, "final": True}

    return AgentExecutor()
