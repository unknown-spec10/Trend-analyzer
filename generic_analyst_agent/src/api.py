"""HTTP API for the Root Cause Analyst agent (FastAPI).

Endpoints:
- GET /health: liveness probe
- POST /analyze: run the agent on a question and return structured results

Run locally:
  uvicorn generic_analyst_agent.src.api:app --reload --port 8000

"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import time
import io

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure env vars are loaded on import
from . import config  # noqa: F401
from .data_source import PandasCSVDataSource, DataFrameDataSource
from .tools import DataQueryTool, search_for_probable_causes, summarize_text
from .agent_graph import create_agent_executor


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


class AnalyzeRequest(BaseModel):
    question: str = Field(..., description="User's question to analyze")
    show_context: bool = Field(False, description="Include internal/external context in response")


class AnalyzeResponse(BaseModel):
    final_answer: Optional[str]
    internal_fact: Optional[Any]
    internal_context: Optional[str]
    external_context: Optional[str]
    external_relevance: Optional[float]
    elapsed_ms: float


def build_agent():
    data_path = project_root() / "data" / "sample_claims.csv"
    data_source = PandasCSVDataSource(
        data_path,
        cache_parquet=True,
        # Narrow to common columns where possible; harmless if columns differ
        usecols=None,
    )
    data_query_tool = DataQueryTool(data_source)
    tools = [data_query_tool.query_data, search_for_probable_causes, summarize_text]
    return create_agent_executor(tools)


app = FastAPI(title="Root Cause Analyst API", version="0.1.0")

# CORS for local dev and simple frontends; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent once at startup
_AGENT = build_agent()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    state = {"messages": [{"type": "human", "content": req.question.strip()}]}
    t0 = time.perf_counter()
    try:
        # Use invoke for a single, consolidated response
        result = _AGENT.invoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent invocation failed: {e}") from e
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Extract structured fields; messages are omitted to keep payload small
    payload = {
        "final_answer": result.get("final_answer"),
        "internal_fact": result.get("internal_fact") if req.show_context else None,
        "internal_context": result.get("internal_context") if req.show_context else None,
        "external_context": result.get("external_context") if req.show_context else None,
        "external_relevance": result.get("external_relevance") if req.show_context else None,
        "elapsed_ms": elapsed_ms,
    }
    return AnalyzeResponse(**payload)


@app.post("/analyze_upload", response_model=AnalyzeResponse)
def analyze_upload(
    file: UploadFile = File(...),
    question: str = Form(...),
    show_context: bool = Form(False),
) -> AnalyzeResponse:
    """Analyze a question using an uploaded CSV as the internal knowledge base.

    This endpoint is convenient for frontends that want to send a CSV per request.
    """
    try:
        raw = file.file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}") from e

    # Build a temporary agent for this request using DataFrame
    ds = DataFrameDataSource(df)
    dq = DataQueryTool(ds)
    tools = [dq.query_data, search_for_probable_causes, summarize_text]
    agent = create_agent_executor(tools)

    state = {"messages": [{"type": "human", "content": question.strip()}]}
    t0 = time.perf_counter()
    try:
        result = agent.invoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent invocation failed: {e}") from e
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    payload = {
        "final_answer": result.get("final_answer"),
        "internal_fact": result.get("internal_fact") if show_context else None,
        "internal_context": result.get("internal_context") if show_context else None,
        "external_context": result.get("external_context") if show_context else None,
        "external_relevance": result.get("external_relevance") if show_context else None,
        "elapsed_ms": elapsed_ms,
    }
    return AnalyzeResponse(**payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("generic_analyst_agent.src.api:app", host="0.0.0.0", port=8000, reload=True)
