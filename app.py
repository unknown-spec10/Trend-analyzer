"""Streamlit UI for the Root Cause Analyst.

Usage:
  streamlit run app.py

This UI lets you:
- Upload a CSV to serve as the internal knowledge base
- Ask questions in a chat and receive concise, cited insights
"""
from __future__ import annotations

import io
from typing import Any, List
import re

import pandas as pd
import streamlit as st

# MUST be called first before any other streamlit commands or imports that use streamlit
st.set_page_config(page_title="Trend Analyzer", layout="wide")

# Import after set_page_config to avoid conflicts
from generic_analyst_agent.src import config  # noqa: F401
from generic_analyst_agent.src.data_source import DataFrameDataSource, create_optimized_csv_data_source
from generic_analyst_agent.src.tools import DataQueryTool, search_for_probable_causes, summarize_text
from generic_analyst_agent.src.agent_graph import create_agent_executor
from generic_analyst_agent.src.session_cache import get_cache, generate_dataset_signature

st.title("Trend Analyzer")

# Session state - Initialize FIRST before any access
if "df" not in st.session_state:
    st.session_state.df = None
if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "history" not in st.session_state:
    st.session_state.history = []  # type: ignore[assignment]
if "stats" not in st.session_state:
    st.session_state.stats = None
if "dataset_signature" not in st.session_state:
    st.session_state.dataset_signature = None

# Initialize cache (1 hour TTL)
cache = get_cache(ttl_seconds=3600)


def extract_concise_pandas_query(code: str) -> str | None:
    """Extract a concise single-line pandas expression from generated code.

    Heuristics:
    - Prefer direct expressions that start with df.
    - If an assignment like `x = df[...]`, return the RHS.
    - If multiple groupby min/max/mean on the same group/column are present,
      condense into a single agg([...]).
    """
    if not code:
        return None

    lines = [ln.strip() for ln in code.splitlines()]

    # Collect candidate expressions that involve df
    exprs: list[str] = []
    group_ops: list[tuple[str, str, str]] = []  # (groupby, column, agg)

    assign_re = re.compile(r"^[A-Za-z_]\w*\s*=\s*(df\..+)$")
    expr_re = re.compile(r"^(df\..+)$")
    # df.groupby('group')['col'].min()/max()/mean()
    gb_re = re.compile(r"df\.groupby\(([^)]*)\)\s*\[(['\"])?(?P<col>\w+)\2?\]\.(?P<agg>min|max|mean)\(\)")

    for ln in lines:
        if not ln or ln.startswith('#'):
            continue
        m = assign_re.match(ln)
        if m:
            exprs.append(m.group(1))
        else:
            m2 = expr_re.match(ln)
            if m2:
                exprs.append(m2.group(1))
        m3 = gb_re.search(ln)
        if m3:
            groupby_part = ln[ln.find('df.groupby('): ln.find(')')+1]
            col = m3.group('col')
            agg = m3.group('agg')
            group_ops.append((groupby_part, col, agg))

    # Condense groupby ops into a single agg([...]) if possible
    if group_ops:
        # Group by (groupby_part, col)
        grouped: dict[tuple[str, str], set[str]] = {}
        for g, col, agg in group_ops:
            grouped.setdefault((g, col), set()).add(agg)
        # Choose the group with the most aggs, prefer min/max/mean set
        best_key = None
        best_aggs: set[str] = set()
        for key, aggs in grouped.items():
            if len(aggs) > len(best_aggs):
                best_key = key
                best_aggs = aggs
        if best_key is not None:
            g, col = best_key
            ordered = [a for a in ['min', 'max', 'mean'] if a in best_aggs]
            if ordered:
                # Build concise expression
                return f"{g}['{col}'].agg({ordered})"

    # Fallback: choose the longest df expression that looks aggregative
    if exprs:
        # Prefer expressions containing groupby/agg/mean/sum/min/max
        def score(e: str) -> tuple[int, int]:
            keywords = sum(1 for k in ['groupby', 'agg', 'mean', 'sum', 'min', 'max', 'count'] if k in e)
            return (keywords, len(e))
        exprs_sorted = sorted(exprs, key=score, reverse=True)
        return exprs_sorted[0]

    return None

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"]) 
    show_context = st.checkbox("Show internal/external context", value=False)
    st.caption("Tip: context shows intermediate facts and web summary.")
    st.markdown("---")
    
    # Question suggestions (show in sidebar after file is loaded)
    if st.session_state.df is not None:
        st.header("💡 Suggested Questions")
        if "suggestions" not in st.session_state:
            with st.spinner("Generating suggestions..."):
                from generic_analyst_agent.src.question_suggester import get_question_suggestions
                try:
                    st.session_state.suggestions = get_question_suggestions(
                        st.session_state.df,
                        st.session_state.stats
                    )
                except Exception as e:
                    st.error(f"Failed to generate suggestions: {e}")
                    st.session_state.suggestions = []
        
        # Display suggestions as clickable buttons
        if "suggestions" in st.session_state and st.session_state.suggestions:
            for idx, question in enumerate(st.session_state.suggestions):
                if st.button(question, key=f"suggestion_{idx}", use_container_width=True):
                    # Set the question and trigger ask
                    st.session_state.pending_question = question
                    st.rerun()

# On upload: read bytes once, parse for preview, build agent
if uploaded is not None:
    try:
        file_bytes = uploaded.read()
        st.session_state.uploaded_bytes = file_bytes
        st.session_state.uploaded_name = uploaded.name or "data.csv"
        
        # Use optimized CSV processor via factory function
        with st.spinner("Processing CSV (optimizing data types and creating cache)..."):
            ds = create_optimized_csv_data_source(file_bytes, force_reload=False)
            df = ds.get_data()
            stats = ds.get_statistics()
            st.session_state.df = df
            st.session_state.stats = stats
            st.session_state.dataset_signature = generate_dataset_signature(df)
            
            # Clear cache when new dataset is loaded
            cache.clear()
            if "suggestions" in st.session_state:
                del st.session_state.suggestions
        
        # Build agent with optimized data source
        dq = DataQueryTool(ds)
        tools = [dq.query_data, search_for_probable_causes, summarize_text]
        st.session_state.agent = create_agent_executor(tools)
        
        memory_mb = stats.get('memory_usage_mb', 0) if stats else 0
        st.success(f"✅ Loaded and optimized CSV: {df.shape[0]:,} rows, {df.shape[1]} columns ({memory_mb:.1f} MB in memory)")
        
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("📊 Data Preview (first 10 rows)"):
                st.dataframe(df.head(10))
        with col2:
            if stats:
                with st.expander("📈 Quick Statistics"):
                    st.json({
                        "Total Rows": f"{stats['row_count']:,}",
                        "Total Columns": stats['column_count'],
                        "Memory Usage": f"{memory_mb:.2f} MB",
                        "Null Values": sum(col_stats['null_count'] for col_stats in stats['columns'].values()),
                    })
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())

# Chat UI
st.subheader("Chat")
chat_container = st.container()

# Summary table of past questions & concise pandas queries (only when context is shown)
if show_context and st.session_state.history:
    st.markdown("### 🧾 Pandas Query Log")
    # Build a compact table
    data = []
    for i, h in enumerate(st.session_state.history, start=1):
        if h.get("pandas_query"):
            data.append({"#": f"Q{i}", "Question": h.get("question", ""), "Pandas Query": h.get("pandas_query")})
    if data:
        import pandas as _pd
        log_df = _pd.DataFrame(data)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

# Check if there's a pending question from suggestion click
if "pending_question" in st.session_state:
    prompt_val = st.session_state.pending_question
    del st.session_state.pending_question
else:
    # Use chat_input for better chat experience
    prompt_val = st.chat_input("Ask a question about your data...", key="chat_input")

# When a question is submitted (or suggestion clicked)
if prompt_val:
    prompt_val = prompt_val.strip()
    if not prompt_val:
        st.warning("Please enter a question.")
    elif st.session_state.agent is None:
        st.warning("Please upload a CSV first.")
    else:
        # Prevent duplicate consecutive entries
        if (
            len(st.session_state.history) > 0
            and st.session_state.history[-1].get("question") == prompt_val
        ):
            # Ignore duplicate consecutive question
            pass
        else:
            # Check cache first
            dataset_sig = st.session_state.dataset_signature or "unknown"
            cached_result = cache.get(prompt_val, dataset_sig)
            
            if cached_result is not None:
                # Use cached result
                st.info("💾 Using cached result")
                result = cached_result
            else:
                # Run agent with progress indicator
                with st.spinner("Analyzing..."):
                    try:
                        # Build state with conversation context
                        messages: List[Any] = [{"type": "human", "content": prompt_val}]
                        
                        # Build conversation history for context
                        conversation_history = []
                        for h in st.session_state.history:
                            conversation_history.append({
                                "question": h["question"],
                                "answer": h.get("final_answer", ""),
                                "fact": h.get("internal_fact"),
                            })
                        
                        state = {
                            "messages": messages,
                            "conversation_history": conversation_history,
                        }
                        result = st.session_state.agent.invoke(state)
                        
                        # Cache the result
                        cache.set(prompt_val, dataset_sig, result)
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        import traceback
                        with st.expander("Error details"):
                            st.code(traceback.format_exc())
                        result = None
            
            if result is not None:
                # Capture generated pandas code from internal_fact
                # The tool returns a dict with generated_code key
                generated_code = None
                internal_fact_obj = result.get("internal_fact")
                if isinstance(internal_fact_obj, dict):
                    generated_code = internal_fact_obj.get("generated_code")

                record = {
                    "question": prompt_val,
                    "final_answer": result.get("final_answer"),
                    "internal_fact": result.get("internal_fact"),  # Always capture for code extraction
                    "internal_context": result.get("internal_context") if show_context else None,
                    "external_context": result.get("external_context") if show_context else None,
                    "sources": result.get("sources"),  # Always capture sources
                    "generated_code": generated_code,  # Store separately for easy access
                    "pandas_query": extract_concise_pandas_query(generated_code) if generated_code else None,
                }
                if not (
                    len(st.session_state.history) > 0
                    and st.session_state.history[-1].get("question") == record["question"]
                    and st.session_state.history[-1].get("final_answer") == record["final_answer"]
                ):
                    st.session_state.history.append(record)

# Render history
with chat_container:
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn.get("question", ""))
        with st.chat_message("assistant"):
            # Show relevant internal context (columns, types, sample) if available
            if show_context and turn.get("internal_context"):
                st.markdown("**📊 Dataset Context**")
                st.write(turn.get("internal_context"))
            
            # Show concise pandas query if available and context is enabled
            if show_context and turn.get("pandas_query"):
                st.markdown("**🐼 Pandas Query**")
                st.code(turn.get("pandas_query"), language="python")
            elif not show_context and turn.get("pandas_query"):
                # Provide a subtle hint when code exists but context is off
                st.caption("💡 Tip: Turn on 'Show internal/external context' in the sidebar to view the Pandas Query.")
            
            # Show full generated pandas code in expander (if available and context is on)
            if show_context and turn.get("generated_code"):
                with st.expander("🧪 View Full Generated Code", expanded=False):
                    st.code(turn.get("generated_code"), language="python")
            if show_context and turn.get("external_context"):
                st.markdown("**External Context**")
                st.write(turn.get("external_context"))
            st.markdown("**Answer**")
            st.write(turn.get("final_answer", ""))
            
            # Display sources if available
            sources = turn.get("sources")
            if sources and isinstance(sources, list) and len(sources) > 0:
                st.markdown("**Sources**")
                for idx, source in enumerate(sources, 1):
                    title = source.get("title", "Untitled")
                    url = source.get("url", "")
                    if url:
                        st.markdown(f"{idx}. [{title}]({url})")
                    else:
                        st.markdown(f"{idx}. {title}")
