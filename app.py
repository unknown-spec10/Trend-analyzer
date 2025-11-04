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

import pandas as pd
import streamlit as st

# MUST be called first before any other streamlit commands or imports that use streamlit
st.set_page_config(page_title="Trend Analyzer", layout="wide")

# Import after set_page_config to avoid conflicts
from generic_analyst_agent.src import config  # noqa: F401
from generic_analyst_agent.src.data_source import DataFrameDataSource, create_optimized_csv_data_source
from generic_analyst_agent.src.tools import DataQueryTool, search_for_probable_causes, summarize_text
from generic_analyst_agent.src.agent_graph import create_agent_executor

st.title("Trend Analyzer")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"]) 
    show_context = st.checkbox("Show internal/external context", value=False)
    st.caption("Tip: context shows intermediate facts and web summary.")
    st.markdown("---")

# Session state
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

with st.form("qa_form", clear_on_submit=True):
    prompt_val = st.text_input(
        "Your question",
        placeholder="Why did claims spike in March 2025?",
        key="prompt_input",
    )
    ask = st.form_submit_button("Ask")

# When Ask is clicked, run agent locally
if ask:
    prompt_val = (prompt_val or "").strip()
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
            # Run agent with progress indicator
            with st.spinner("Analyzing..."):
                try:
                    # Build state with conversation context
                    messages: List[Any] = []
                    for h in st.session_state.history:
                        messages.append({"type": "human", "content": h["question"]})
                    messages.append({"type": "human", "content": prompt_val})
                    
                    state = {"messages": messages}
                    result = st.session_state.agent.invoke(state)
                    
                    record = {
                        "question": prompt_val,
                        "final_answer": result.get("final_answer"),
                        "internal_fact": result.get("internal_fact") if show_context else None,
                        "internal_context": result.get("internal_context") if show_context else None,
                        "external_context": result.get("external_context") if show_context else None,
                        "sources": result.get("sources"),  # Always capture sources
                    }
                    if not (
                        len(st.session_state.history) > 0
                        and st.session_state.history[-1].get("question") == record["question"]
                        and st.session_state.history[-1].get("final_answer") == record["final_answer"]
                    ):
                        st.session_state.history.append(record)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

# Render history
with chat_container:
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn.get("question", ""))
        with st.chat_message("assistant"):
            if show_context and turn.get("internal_fact") is not None:
                st.markdown("**Internal Fact**")
                st.json(turn.get("internal_fact"))
                if turn.get("internal_context"):
                    st.markdown("**Internal Context**")
                    st.write(turn.get("internal_context"))
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
