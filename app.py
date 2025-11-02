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

import requests
import pandas as pd
import streamlit as st

# Ensure env vars are loaded (keys, etc.)
from generic_analyst_agent.src import config  # noqa: F401

st.set_page_config(page_title="Root Cause Analyst", layout="wide")

st.title("Root Cause Analyst")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"]) 
    show_context = st.checkbox("Show internal/external context", value=False)
    st.caption("Tip: context shows intermediate facts and web summary.")
    api_url = st.text_input("API base URL", value="http://localhost:8000", help="FastAPI server base URL")

# Session state
if "df" not in st.session_state:
    st.session_state.df = None
if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "history" not in st.session_state:
    st.session_state.history = []  # type: ignore[assignment]

# On upload: read bytes once, parse for preview, persist bytes for API call
if uploaded is not None:
    try:
        file_bytes = uploaded.read()
        st.session_state.uploaded_bytes = file_bytes
        st.session_state.uploaded_name = uploaded.name or "data.csv"
        df = pd.read_csv(io.BytesIO(file_bytes))
        st.session_state.df = df
        st.success(f"Loaded CSV with {df.shape[0]:,} rows and {df.shape[1]} columns")
        with st.expander("Preview (first 10 rows)"):
            st.dataframe(df.head(10))
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")

# Chat UI
st.subheader("Chat")
chat_container = st.container()

with st.form("qa_form", clear_on_submit=True):
    prompt_val = st.text_input(
        "Your question",
        placeholder="Why did Electronics claims spike in March 2025?",
        key="prompt_input",
    )
    ask = st.form_submit_button("Ask")

# When Ask is clicked, call backend API
if ask:
    prompt_val = (prompt_val or "").strip()
    if not prompt_val:
        st.warning("Please enter a question.")
    elif st.session_state.uploaded_bytes is None:
        st.warning("Please upload a CSV first.")
    else:
        # Prevent duplicate consecutive entries
        if (
            len(st.session_state.history) > 0
            and st.session_state.history[-1].get("question") == prompt_val
        ):
            # Ignore duplicate consecutive question
            pass

        try:
            files = {
                "file": (
                    st.session_state.uploaded_name or "data.csv",
                    io.BytesIO(st.session_state.uploaded_bytes),
                    "text/csv",
                )
            }
            form = {
                "question": prompt_val,
                "show_context": str(show_context).lower(),
            }
            resp = requests.post(
                f"{api_url.rstrip('/')}/analyze_upload",
                files=files,
                data=form,
                timeout=300,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
            result = resp.json()
        except Exception as e:
            st.error(f"API request failed: {e}")
            st.stop()

        record = {
            "question": prompt_val,
            "final_answer": result.get("final_answer"),
            "internal_fact": result.get("internal_fact") if show_context else None,
            "internal_context": result.get("internal_context") if show_context else None,
            "external_context": result.get("external_context") if show_context else None,
        }
        if not (
            len(st.session_state.history) > 0
            and st.session_state.history[-1].get("question") == record["question"]
            and st.session_state.history[-1].get("final_answer") == record["final_answer"]
        ):
            st.session_state.history.append(record)
        # Form has clear_on_submit=True; no manual clearing or rerun required

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
