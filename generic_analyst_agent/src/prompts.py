"""Prompts and system persona for the Root Cause Analyst agent (SRP)."""


def _generate_adaptive_example(numeric_cols: list, categorical_cols: list, date_cols: list) -> str:
    """Generate example code based on actual column types available in the dataset."""
    
    if not numeric_cols and not categorical_cols:
        return """EXAMPLE (basic count):
result = {{
    "metric": "record_count",
    "value": len(df),
    "period": "full_dataset",
    "segment": "all",
    "unit": "records",
    "details": {{}},
    "summary": f"Dataset contains {{len(df)}} records"
}}
print(json.dumps(result))"""
    
    # Pick first available numeric and categorical columns for example
    num_col = numeric_cols[0] if numeric_cols else None
    cat_col = categorical_cols[0] if categorical_cols else None
    
    if num_col and cat_col:
        return f"""EXAMPLE (using YOUR columns: {num_col}, {cat_col}):
# For question like "What is average {num_col} for each {cat_col}?"
by_group = df.groupby('{cat_col}')['{num_col}'].mean()
overall = df['{num_col}'].mean()

result = {{
    "metric": "average_{num_col}_by_{cat_col}",
    "value": float(overall),
    "period": "full_dataset",
    "segment": "all_{cat_col}s",
    "unit": "appropriate_unit",
    "details": {{
        "by_{cat_col}": by_group.to_dict()
    }},
    "summary": f"Average {num_col} is {{overall:.2f}}. By {cat_col}: " + ", ".join([f"{{k}}: {{v:.2f}}" for k,v in by_group.items()])
}}
print(json.dumps(result))"""
    
    elif num_col:
        return f"""EXAMPLE (using YOUR numeric column: {num_col}):
result_value = df['{num_col}'].mean()
result = {{
    "metric": "average_{num_col}",
    "value": float(result_value),
    "period": "full_dataset",
    "segment": "all",
    "unit": "appropriate_unit",
    "details": {{}},
    "summary": f"Average {num_col} is {{result_value:.2f}}"
}}
print(json.dumps(result))"""
    
    else:  # Only categorical
        return f"""EXAMPLE (using YOUR categorical column: {cat_col}):
counts = df['{cat_col}'].value_counts()
result = {{
    "metric": "{cat_col}_distribution",
    "value": int(counts.max()),
    "period": "full_dataset",
    "segment": counts.idxmax(),
    "unit": "records",
    "details": {{"by_{cat_col}": counts.to_dict()}},
    "summary": f"Most common {cat_col}: {{counts.idxmax()}} ({{counts.max()}} records)"
}}
print(json.dumps(result))"""


ROOT_CAUSE_ANALYST_PROMPT = (
    "You are Root Cause Analyst, a disciplined analyst that explains WHY things happen by "
    "following a strict three-step procedure.\n\n"
    "Mission:\n"
    "1) Internal Data Analysis: Use the query_data tool FIRST to extract a single, precise, quantitative fact from the local dataset.\n"
    "2) External Web Research: Use the search_for_probable_causes tool SECOND to find credible, real-world causes that explain that fact.\n"
    "3) Synthesis: Combine the internal fact and the external context into a coherent, defensible explanation with sources.\n\n"
    "Rules:\n"
    "- Always call query_data before search_for_probable_causes.\n"
    "- The internal fact must be specific (e.g., exact counts, amounts, deltas, dates, locations).\n"
    "- The web research should prefer objective sources (news reports, government data, industry analyses).\n"
    "- Be transparent about uncertainty and offer 1-2 alternative hypotheses if applicable.\n"
    "- Final answer format:\n"
    "  Internal Fact: <one sentence fact>\n"
    "  Likely Causes: <synthesized explanation with brief citations>\n"
    "  Confidence: <low/medium/high with 1-2 sentence justification>\n"
)


# Data Analysis Prompt - Used by DataQueryTool in tools.py
def get_data_analysis_prompt(user_query: str, schema_info: str, head_str: str, 
                              numeric_cols: list, categorical_cols: list, date_cols: list) -> str:
    """Generate ADAPTIVE prompt for data analysis code generation that works with ANY CSV structure."""
    
    # Dynamically determine what types of analysis are possible
    has_numeric = len(numeric_cols) > 0
    has_categorical = len(categorical_cols) > 0
    has_dates = len(date_cols) > 0
    
    # Build column-specific guidance
    column_guidance = []
    if has_numeric:
        column_guidance.append(f"- You can aggregate numeric columns: {', '.join(numeric_cols[:5])}")
    if has_categorical:
        column_guidance.append(f"- You can group by categorical columns: {', '.join(categorical_cols[:5])}")
    if has_dates:
        column_guidance.append(f"- You can analyze trends over time using: {', '.join(date_cols[:3])}")
    
    # Determine appropriate unit based on data (smart inference)
    sample_numeric = numeric_cols[0] if has_numeric else None
    unit_hint = "records"  # Default
    if sample_numeric:
        col_lower = sample_numeric.lower()
        if any(term in col_lower for term in ['price', 'cost', 'revenue', 'amount', 'claim', 'salary', 'payment', 'fee', 'charge']):
            unit_hint = "currency (USD/EUR/etc.)"
        elif any(term in col_lower for term in ['temp', 'temperature', 'celsius', 'fahrenheit']):
            unit_hint = "temperature units"
        elif any(term in col_lower for term in ['count', 'quantity', 'number', 'total']):
            unit_hint = "count"
        elif any(term in col_lower for term in ['percent', 'rate', 'ratio', 'proportion']):
            unit_hint = "percentage"
        elif any(term in col_lower for term in ['weight', 'mass', 'kg', 'lb']):
            unit_hint = "weight units"
        elif any(term in col_lower for term in ['distance', 'length', 'km', 'mile']):
            unit_hint = "distance units"
    
    # Generate adaptive examples based on actual columns
    example_code = _generate_adaptive_example(numeric_cols, categorical_cols, date_cols)
    
    return f"""You are an expert data analyst. The DataFrame `df` is ALREADY LOADED and ready to use.

CRITICAL REQUIREMENTS:
1. FIRST check if the question's key terms match the ACTUAL columns in the schema
2. If the question asks about columns/concepts NOT in the dataset, return an error result
3. Analyze the ACTUAL columns to understand what analysis is possible
4. Answer using ONLY the data present in `df`
5. Build a STRUCTURED dictionary named `result` with these exact keys:
   {{"metric": str, "value": number, "period": str, "segment": str, "unit": str, "details": dict, "summary": str}}

ERROR HANDLING:
If the question doesn't match the data, return:
{{
    "metric": "error",
    "value": None,
    "period": "unknown",
    "segment": "unknown",
    "unit": "unknown",
    "details": {{"error": "Question asks about data not present in this dataset"}},
    "summary": "Unable to answer: question asks about data not in this dataset"
}}

DATASET ANALYSIS CAPABILITIES:
{chr(10).join(column_guidance) if column_guidance else "- This dataset has limited analysis options"}

FIELD SPECIFICATIONS (adapt to YOUR data):
- `metric`: Short name describing what you measured (use terms from the actual columns)
- `value`: PRIMARY numeric answer - the MAIN result (use None if not applicable)
- `period`: If you have date columns: time window; otherwise: "full_dataset"
- `segment`: The primary grouping from YOUR categorical columns (or "all" if no grouping)
  * Inspect the user's question - if they ask about a category, use the group with highest value
  * If they ask about multiple dimensions, pick the MOST relevant one as segment
- `unit`: Appropriate unit from YOUR data context (suggested: {unit_hint})
- `details`: Dict with secondary breakdowns using YOUR column names
  * For multi-part questions, put the answer to the secondary part here
  * For "for each X" questions, put per-group breakdown in details as {{"by_X": {{group1: value1, group2: value2}}}}
  * Example: "average claim for each region" → details should have {{"by_region": {{"northwest": 1234, "southeast": 5678}}}}
  * NEVER use dimension values as nested keys - keep structure flat
- `summary`: Plain English answer with actual numbers from YOUR calculation

⚠️ CRITICAL - MULTI-METRIC QUESTIONS (READ CAREFULLY):
When user asks for MULTIPLE metrics in ONE question (e.g., "min, max, and average" OR "minimum, maximum, and mean"):
YOU MUST CALCULATE **ALL** REQUESTED METRICS - DO NOT CALCULATE ONLY ONE!

Steps:
1. Parse the question to identify ALL metrics requested (min, max, mean, sum, count, median, std, etc.)
2. Use .agg([list_of_all_metrics]) to calculate them ALL at once
3. Store ALL results in the response details

Pattern: "What are [metric1], [metric2], and [metric3] of [column] by [groupby_column]?"
✅ CORRECT approach:
- If grouping: stats = df.groupby('groupby_col')['numeric_col'].agg(['metric1', 'metric2', 'metric3'])
- If overall: Store each metric separately in details
- Put ALL requested metrics in details with clear keys like "by_category_min", "by_category_max", "by_category_mean"
- Include overall statistics too: "overall_min", "overall_max", "overall_mean"

❌ WRONG - Do NOT calculate only one metric when multiple are requested!

IMPORTANT - "FOR EACH" QUESTIONS:
When user asks "for each [category]" or "by [category]", they want per-group results:
- Put the primary answer (e.g., average across all groups) in `value`
- Put the per-group breakdown in `details` under key "by_[category]"
- Include ALL groups in the breakdown, not just top/bottom
- Example: "average claim for each region" should have details={{"by_region": {{"northwest": 1234, "southeast": 5678, ...}}}}

AVAILABLE COLUMNS IN THIS DATASET:
- Numeric: {numeric_cols if has_numeric else "None"}
- Categorical: {categorical_cols if has_categorical else "None"}
- Dates: {date_cols if has_dates else "None"}

- Build a dict named `result` with ALL required keys (metric, value, period, segment, unit, details, summary)
- At the end, print ONLY: print(json.dumps(result))
- Do NOT import anything (json, pd, int, float, str, dict already available)
- Do NOT use print for anything else (no debug prints)
- Do NOT define functions - write direct pandas operations
- Do NOT write comments, explanations, or markdown
- Use df.columns to verify column names before filtering
- Handle missing values gracefully with .dropna() or .fillna()
- Ensure `value` is numeric (use int() or float() to convert) or None if not applicable
- Output ONLY executable Python code (no ```python fences, no explanations)
- Keep code SIMPLE and DIRECT - no complex functions or if-elif chains

{example_code}

User question:
{user_query}

DataFrame schema:
{schema_info}

Data sample (first 10 rows):
{head_str}
"""


# Search Query Generation Prompt - Used in agent_graph.py
def get_search_query_prompt(metric: str, segment: str, period: str, summary: str) -> str:
    """Generate prompt for creating web search query."""
    return f"""Generate a focused web search query to find external factors explaining this data pattern.

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


# Context Summarization Prompt - Used in agent_graph.py
def get_context_summary_prompt(text: str) -> str:
    """Generate prompt for summarizing external context."""
    return f"""Summarize the CONTEXT into tight bullets first, then one short wrap-up line.
Rules:
- 5-7 bullets, each <= 18 words.
- Max 1200 characters total.
- Preserve any inline citations like (Title - URL) if present.
- Focus on causal drivers and evidence.

CONTEXT:
{text}
"""


# Relevance Scoring Prompt - Used in agent_graph.py
def get_relevance_score_prompt(internal_fact_brief: str, context: str) -> str:
    """Generate prompt for scoring relevance of external context."""
    return f"""Score how relevant the CONTEXT is to the INTERNAL FACT on a 0.0-1.0 scale.
Return ONLY a number with up to 2 decimals, no text.

INTERNAL FACT: {internal_fact_brief}

CONTEXT:
{context}

Score:"""


# Question Expansion Prompt - Used in agent_graph.py
def get_question_expansion_prompt(prev_question: str, prev_answer: str, current_question: str) -> str:
    """Generate prompt for expanding follow-up questions with context."""
    return f"""Expand this follow-up question using the previous conversation context.

PREVIOUS QUESTION:
{prev_question}

PREVIOUS ANSWER:
{prev_answer[:500]}

CURRENT FOLLOW-UP:
{current_question}

Task: Rewrite the follow-up as a complete, standalone question that includes the necessary context.

Example 1:
Previous: "Which region claimed the most?"
Follow-up: "What about females?"
Expanded: "What is the total claimed by females in the region that claimed the most?"

Example 2:
Previous: "What are the top 3 products by sales?"
Follow-up: "How about last year?"
Expanded: "What are the top 3 products by sales for last year?"

Respond with ONLY the expanded question, no explanation.
"""


# Search Decision Prompt - Used in agent_graph.py
def get_search_decision_prompt(question: str, internal_summary: str, internal_struct: dict) -> str:
    """Generate prompt for deciding if external search is needed."""
    import json
    return f"""You are analyzing whether a data-driven answer is sufficient or needs external research.

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


# Question Suggestions Prompt - Used in question_suggester.py
def get_question_suggestions_prompt(context: str, max_questions: int) -> str:
    """Generate prompt for suggesting questions based on dataset."""
    return f"""You are an expert data analyst. Generate {max_questions} insightful questions that a user could ask about this dataset.

DATASET CHARACTERISTICS:
{context}

REQUIREMENTS:
1. Questions should be specific and actionable (not generic like "Show me the data")
2. Mix different question types:
   - Aggregation (total, average, sum, count)
   - Comparison (which X has the highest Y?)
   - Segmentation (breakdown by category)
   - Trend analysis (if date columns exist)
   - Outlier detection (if numeric columns exist)
3. Use actual column names from the dataset
4. Questions should be 8-15 words long
5. Avoid yes/no questions

Return ONLY the questions, one per line, numbered 1-{max_questions}.
"""
