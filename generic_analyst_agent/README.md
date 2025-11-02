# Root Cause Analyst (LangGraph + LangChain)

A small, SOLID-architected Python project that answers complex "why" questions by:

1) Internal Data Analysis (local CSV) → extract a single quantitative fact
2) External Web Research → search the web for real-world causes
3) Synthesis → combine both into a concise, sourced explanation

The code separates concerns cleanly (SRP) and depends on abstractions (DIP) so you can swap data sources and models without breaking higher-level logic.

## What’s inside

- `src/config.py` — loads configuration/API keys from `.env`
- `src/data_source.py` — `BaseDataSource` abstraction + `PandasCSVDataSource`
- `src/tools.py` — `DataQueryTool` (LLM → pandas code) and `search_for_probable_causes` (web search)
- `src/prompts.py` — system prompt for the Root Cause Analyst persona
- `src/agent_graph.py` — orchestration graph (LangGraph) with strict order: data → search → synthesize
- `src/main.py` — runnable example wiring everything together
- `data/sample_claims.csv` — demo dataset with an intentional March 2025 spike in Houston Electronics

## Quickstart

1) Create/activate a virtual environment, then install deps:

```powershell
# from the project root
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

2) Configure your API keys in `.env`:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

3) Run the demo:

```powershell
python -m generic_analyst_agent.src.main
```

You should see the internal fact detected from the CSV, relevant external context, and a synthesized final answer.

## Notes on data

- You can replace `data/sample_claims.csv` with any CSV that fits your domain. The agent is schema-aware and will infer column types. No need to worry about pre-generating more CSVs—start with the included sample and iterate.

## Search and LLM providers

This implementation uses:
- Groq via `langchain-groq` (model: `llama-3.1-8b-instant`) for internal data reasoning and final synthesis
- Google Custom Search JSON API (`google-api-python-client`) to retrieve relevant pages
- Google Gemini 2.5 Flash (`google-generativeai`) to synthesize a concise, cited explanation from the retrieved snippets

Environment variables required:

```
GROQ_API_KEY=...
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...   # your Custom Search Engine ID (cx)
GEMINI_API_KEY=...
```

The codebase follows SOLID principles and industry-grade structure, so swapping providers (e.g., to OpenAI/Anthropic or other search APIs) remains localized to tools and config.

## Architecture and principles

- SRP (Single Responsibility Principle)
  - Each module has one job (config, data access, tools, prompts, orchestration, entrypoint)
- DIP (Dependency Inversion Principle)
  - `DataQueryTool` depends on `BaseDataSource` interface, not a concrete CSV reader
  - LLM and search providers are injected or localized, so high-level logic remains stable

## Safety disclaimer

`DataQueryTool` uses `exec` to run model-generated pandas code for demonstration purposes only. In production, you must sandbox aggressively (e.g., process isolation, time/memory limits, allow-listed APIs) and add robust validation.

## Troubleshooting

- If you see import errors for `langgraph`, `langchain`, `pandas`, or `tavily`, ensure you installed packages into the active venv: `pip install -r requirements.txt`.
- If LLM or search calls fail, double-check your `.env` keys and network access.
