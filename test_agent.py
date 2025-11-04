"""Quick test script to see how the agent handles the medical insurance dataset."""
import logging
import sys

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

import pandas as pd
from generic_analyst_agent.src.data_source import DataFrameDataSource
from generic_analyst_agent.src.tools import DataQueryTool, search_for_probable_causes, summarize_text
from generic_analyst_agent.src.agent_graph import create_agent_executor

# Load the test CSV
print("=" * 80)
print("Loading test dataset...")
df = pd.read_csv("insurance_data - insurance_data.csv")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head(3))
print("=" * 80)

# Create the agent
print("\nBuilding agent...")
ds = DataFrameDataSource(df)
dq = DataQueryTool(ds)
tools = [dq.query_data, search_for_probable_causes, summarize_text]
agent = create_agent_executor(tools)

# Test question
question = "Which region has claimed most and what are the possible causes for that?"
print(f"\nQUESTION: {question}")
print("=" * 80)

# Run the agent
print("\nRunning agent...\n")
state = {"messages": [{"type": "human", "content": question}]}
result = agent.invoke(state)

# Print results
print("\n" + "=" * 80)
print("RESULTS:")
print("=" * 80)
print(f"\nInternal Fact: {result.get('internal_fact')}")
print(f"\nInternal Context:\n{result.get('internal_context')}")
print(f"\nExternal Context:\n{result.get('external_context', 'None')}")
print(f"\nFinal Answer:\n{result.get('final_answer')}")
print("=" * 80)
