"""Prompts and system persona for the Root Cause Analyst agent (SRP)."""

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
    """Generate prompt for data analysis code generation."""
    return f"""You are an expert data analyst. Given a pandas DataFrame named `df`, write Python code that:

CRITICAL REQUIREMENTS:
1. FIRST check if the question's key terms match the ACTUAL columns in the schema
2. If the question asks about columns/concepts NOT in the dataset, return an error result
3. Analyze the ACTUAL columns in the schema to understand the dataset domain
4. Answer the user's question using ONLY the data present in `df`
5. Build a STRUCTURED dictionary named `result` with these exact keys:
   {{"metric": str, "value": number, "period": str, "segment": str, "unit": str, "details": dict, "summary": str}}

ERROR HANDLING FOR MISMATCHED QUESTIONS:
If the user asks about data that doesn't exist (e.g., "stock price" when there's no stock column), return:
{{
    "metric": "error",
    "value": None,
    "period": "unknown",
    "segment": "unknown",
    "unit": "unknown",
    "details": {{}},
    "summary": "Unable to answer: question asks about data not present in this dataset"
}}

FIELD SPECIFICATIONS:
- `metric`: Short name describing what was measured (e.g., "regional_distribution", "gender_breakdown")
- `value`: PRIMARY numeric answer (count, sum, average, percentage) - this is the MAIN result
- `period`: Time window if dates exist; otherwise "full_dataset" or describe the data scope
- `segment`: The primary grouping/category/demographic that answers the question
  * Inspect the user's question - if they ask about "region", use the region with highest value
  * If they ask about "gender" or "sex", use the dominant gender
  * If they ask about multiple dimensions, pick the MOST relevant one as segment
- `unit`: Appropriate unit ("records", "claims", "patients", "USD", "percentage", etc.)
- `details`: Dict with secondary breakdowns - MUST contain ACTUAL VALUES from the data
  * For multi-part questions, put the answer to the secondary part here
  * Example: If asking "which region claimed most AND which gender in that region", `details` should have {{"by_gender": {{"male": <number>, "female": <number>}}}}
  * NEVER use region names or other dimension values as keys in gender breakdowns
- `summary`: 1-2 sentence plain English answer to the question with ALL key numbers including secondary dimensions

DATASET CONTEXT (adapt your analysis to these columns):
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Date columns: {date_cols}

CODE RULES:
- Build a dict named `result` with ALL required keys (metric, value, period, segment, unit, details, summary)
- At the end, print ONLY: print(json.dumps(result))
- Do NOT import anything (json, pd, int, float, str, dict already available)
- Do NOT use print for anything else (no debug prints)
- Do NOT write comments, explanations, or markdown
- Use df.columns to verify column names before filtering
- Handle missing values gracefully with .dropna() or .fillna()
- Ensure `value` is numeric (use int() or float() to convert)
- Output ONLY executable Python code (no ```python fences, no explanations)

EXAMPLE CODE STRUCTURE:
# Find the answer using pandas
result_value = ...  # your calculation
result = {{
    "metric": "descriptive_name",
    "value": int(result_value),
    "period": "full_dataset",
    "segment": "category_name",
    "unit": "records",
    "details": {{}},
    "summary": "Plain English answer"
}}
print(json.dumps(result))

EXAMPLE FOR MULTI-DIMENSIONAL QUESTIONS:
# Question: "Which region has the most claims and which gender in that region has more?"
# Step 1: Find the region with highest claims
region_totals = df.groupby('region')['claim'].sum()
top_region = region_totals.idxmax()
top_region_value = int(region_totals.max())

# Step 2: Within that region, find gender breakdown
region_data = df[df['region'] == top_region]
gender_totals = region_data.groupby('gender')['claim'].sum()

# Step 3: Build result with BOTH answers
result = {{
    "metric": "regional_claim_analysis",
    "value": top_region_value,
    "period": "full_dataset",
    "segment": top_region,
    "unit": "USD",
    "details": {{
        "by_gender": gender_totals.to_dict(),
        "top_gender": gender_totals.idxmax(),
        "top_gender_amount": int(gender_totals.max())
    }},
    "summary": f"The {{top_region}} region has the highest claims ({{top_region_value}} USD). Within this region, {{gender_totals.idxmax()}} has claimed more ({{int(gender_totals.max())}} USD vs {{int(gender_totals.min())}} USD)."
}}
print(json.dumps(result))

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
