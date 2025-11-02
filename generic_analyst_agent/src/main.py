"""Project entry point demonstrating the Root Cause Analyst agent.

This script wires configuration, data source, tools, agent graph, and runs a sample query.
"""
from __future__ import annotations

from pathlib import Path
import json
import argparse
import logging

# Ensure env vars are loaded on import
from . import config  # noqa: F401  # side-effect: load .env
from .data_source import PandasCSVDataSource
from .tools import DataQueryTool, search_for_probable_causes, summarize_text
from .agent_graph import create_agent_executor


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Root Cause Analyst")
    parser.add_argument("--question", "-q", type=str, help="Question to analyze", default=None)
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", "-Q", action="store_true", help="Reduce logs (ERROR only)")
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default=None,
        help="Override log level",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print internal/external context sections in addition to the final answer",
    )
    args = parser.parse_args()

    # Determine base log level (default WARNING to reduce noise)
    if args.quiet:
        base_level = logging.ERROR
    elif args.log_level is not None:
        base_level = getattr(logging, args.log_level)
    elif args.verbose:
        base_level = logging.DEBUG
    else:
        base_level = logging.WARNING

    logging.basicConfig(
        level=base_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    # Suppress noisy third-party libraries unless verbose explicitly requested
    if base_level > logging.INFO:
        for noisy in ("httpx", "googleapiclient", "googleapiclient.discovery_cache", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    data_path = project_root() / "data" / "sample_claims.csv"

    # Initialize data source and query tool (DIP in action)
    data_source = PandasCSVDataSource(data_path)
    data_query_tool = DataQueryTool(data_source)

    # Tools list given to the agent: instance-bound query_data tool and search function tool
    tools = [data_query_tool.query_data, search_for_probable_causes, summarize_text]

    # Build the agent
    agent = create_agent_executor(tools)

    # Prompt the user for a question
    user_question = args.question or input("Enter your question for the Root Cause Analyst: ")

    # Seed state with initial user message
    state = {"messages": [{"type": "human", "content": user_question}]}

    # Stream values through the strict graph
    logger.info("--- Running Root Cause Analyst ---")
    try:
        seen_internal = set()
        seen_external = set()
        seen_internal_ctx = set()
        # Accumulate latest values for a clean end-of-run printout
        last_internal = None
        last_internal_ctx = None
        last_external = None
        last_final = None
        for update in agent.stream(state, stream_mode="values"):
            # Each update is a partial state; print key events
            internal = update.get("internal_fact")
            internal_ctx = update.get("internal_context")
            external = update.get("external_context")
            final = update.get("final_answer")
            if internal is not None:
                # Normalize to a string for deduplication and display
                if isinstance(internal, dict):
                    internal_norm = json.dumps(internal, sort_keys=True)
                    display_internal = internal
                else:
                    internal_norm = str(internal)
                    display_internal = internal
                if internal_norm not in seen_internal:
                    if args.show_context:
                        print("[Internal Fact]", display_internal)
                    seen_internal.add(internal_norm)
                last_internal = display_internal
            if internal_ctx and internal_ctx not in seen_internal_ctx:
                if args.show_context:
                    print("\n[Internal Context]\n", internal_ctx)
                seen_internal_ctx.add(internal_ctx)
                last_internal_ctx = internal_ctx
            if external and external not in seen_external:
                if args.show_context:
                    print("\n[External Context]\n", external)
                seen_external.add(external)
                last_external = external
            if final:
                if args.show_context:
                    print("\n[Final Answer]\n", final)
                last_final = final
        # If not showing context during streaming, print a clean summary at the end
        if not args.show_context:
            if last_final:
                print(last_final)
            elif last_internal_ctx:
                print(last_internal_ctx)
    except AttributeError:
        # Fallback for older langgraph without stream_mode
        result = agent.invoke(state)
        final_text = result.get("final_answer")
        if args.show_context:
            # Print any context we can recover
            internal = result.get("internal_fact")
            external = result.get("external_context")
            if internal:
                print("[Internal Fact]", internal)
            if external:
                print("\n[External Context]\n", external)
            print("\n[Final Answer]\n", final_text)
        else:
            print(final_text)


if __name__ == "__main__":
    main()
