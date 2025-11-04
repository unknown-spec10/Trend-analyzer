"""Question Suggester: Generate relevant questions based on dataset characteristics.

This module analyzes the dataset's schema, statistics, and data types to automatically
suggest 5-7 high-value questions that users can ask.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List

import pandas as pd
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


class QuestionSuggester:
    """Generates contextual question suggestions based on dataset analysis."""
    
    def __init__(self, llm_model: str = "llama-3.1-8b-instant", temperature: float = 0.5):
        """Initialize the question suggester.
        
        Args:
            llm_model: The LLM model to use for generation
            temperature: Temperature for LLM (higher = more creative)
        """
        self.llm = ChatGroq(model=llm_model, temperature=temperature)
        
    def generate_suggestions(
        self,
        df: pd.DataFrame,
        stats: Dict[str, Any] | None = None,
        max_questions: int = 7,
    ) -> List[str]:
        """Generate question suggestions based on dataset characteristics.
        
        Args:
            df: The dataframe to analyze
            stats: Pre-computed statistics (optional)
            max_questions: Maximum number of questions to generate
            
        Returns:
            List of suggested questions
        """
        try:
            # Analyze schema
            schema_info = self._analyze_schema(df)
            
            # Analyze data characteristics
            data_patterns = self._analyze_patterns(df, stats)
            
            # Generate questions using LLM
            questions = self._generate_with_llm(schema_info, data_patterns, max_questions)
            
            return questions
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            # Return fallback generic questions
            return self._fallback_questions(df)
    
    def _analyze_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the dataset schema to identify key columns."""
        schema = {
            "total_columns": len(df.columns),
            "total_rows": len(df),
            "numeric_columns": [],
            "categorical_columns": [],
            "date_columns": [],
            "high_cardinality": [],  # Good for grouping
            "low_cardinality": [],   # Good for filtering
        }
        
        for col in df.columns:
            dtype = df[col].dtype
            nunique = df[col].nunique()
            
            # Numeric columns (for aggregation)
            if pd.api.types.is_numeric_dtype(dtype):
                schema["numeric_columns"].append({
                    "name": col,
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                })
            
            # Date columns (for time-based analysis)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                schema["date_columns"].append(col)
            
            # Categorical columns
            elif dtype == 'object' or dtype.name == 'category':
                schema["categorical_columns"].append(col)
                
                # High cardinality (good for detailed grouping)
                if nunique > 10 and nunique < len(df) * 0.5:
                    schema["high_cardinality"].append(col)
                
                # Low cardinality (good for segmentation)
                elif nunique <= 10:
                    schema["low_cardinality"].append({
                        "name": col,
                        "values": df[col].value_counts().head(5).to_dict(),
                    })
        
        return schema
    
    def _analyze_patterns(self, df: pd.DataFrame, stats: Dict[str, Any] | None) -> Dict[str, Any]:
        """Analyze data patterns and characteristics."""
        patterns = {
            "has_nulls": df.isnull().sum().sum() > 0,
            "potential_outliers": [],
            "potential_trends": [],
        }
        
        # Check for outliers in numeric columns (simple IQR method)
        for col in df.select_dtypes(include=['number']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                patterns["potential_outliers"].append({
                    "column": col,
                    "count": len(outliers),
                })
        
        # Check for time-based patterns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            patterns["potential_trends"].append("time_series_available")
        
        return patterns
    
    def _generate_with_llm(
        self,
        schema: Dict[str, Any],
        patterns: Dict[str, Any],
        max_questions: int,
    ) -> List[str]:
        """Use LLM to generate contextual questions."""
        # Build context for LLM
        context_parts = []
        
        # Schema context
        context_parts.append(f"Dataset has {schema['total_rows']:,} rows and {schema['total_columns']} columns.")
        
        if schema['numeric_columns']:
            numeric_names = [c['name'] for c in schema['numeric_columns'][:3]]
            context_parts.append(f"Numeric columns for aggregation: {', '.join(numeric_names)}")
        
        if schema['low_cardinality']:
            for cat in schema['low_cardinality'][:2]:
                values = list(cat['values'].keys())[:3]
                context_parts.append(f"Categorical column '{cat['name']}' has values like: {', '.join(map(str, values))}")
        
        if schema['date_columns']:
            context_parts.append(f"Date columns available: {', '.join(schema['date_columns'][:2])}")
        
        # Pattern context
        if patterns['has_nulls']:
            context_parts.append("Dataset has missing values that could be analyzed.")
        
        if patterns['potential_outliers']:
            outlier_cols = [o['column'] for o in patterns['potential_outliers'][:2]]
            context_parts.append(f"Potential outliers detected in: {', '.join(outlier_cols)}")
        
        context = "\n".join(context_parts)
        
        # Generate questions
        prompt = f"""You are an expert data analyst. Generate {max_questions} insightful questions that a user could ask about this dataset.

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
        
        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", str(response))
            
            # Parse questions from response
            questions = []
            for line in content.strip().split('\n'):
                line = line.strip()
                # Remove numbering (1. 2. etc)
                if line and line[0].isdigit():
                    line = line.split('.', 1)[-1].strip()
                if line and len(line) > 10:  # Filter out empty or very short lines
                    questions.append(line)
            
            # Ensure we have at least some questions
            if len(questions) < 3:
                logger.warning("LLM generated too few questions, adding fallbacks")
                questions.extend(self._fallback_questions(None)[:max_questions - len(questions)])
            
            return questions[:max_questions]
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._fallback_questions(None)[:max_questions]
    
    def _fallback_questions(self, df: pd.DataFrame | None) -> List[str]:
        """Fallback generic questions if LLM fails."""
        if df is None:
            return [
                "What are the summary statistics of the dataset?",
                "Which categories have the highest counts?",
                "Are there any missing values in the data?",
                "What is the distribution of numeric values?",
                "Show me the top 5 records by value",
            ]
        
        questions = []
        
        # Generic questions based on column types
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            questions.append(f"What is the total {numeric_cols[0]}?")
            questions.append(f"What is the average {numeric_cols[0]}?")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            questions.append(f"Which {categorical_cols[0]} has the highest {numeric_cols[0]}?")
        
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            questions.append(f"What is the trend of {numeric_cols[0]} over time?")
        
        questions.append("Are there any outliers in the numeric data?")
        questions.append("What is the distribution of values across categories?")
        
        return questions


def get_question_suggestions(df: pd.DataFrame, stats: Dict[str, Any] | None = None) -> List[str]:
    """Convenience function to get question suggestions.
    
    Args:
        df: The dataframe to analyze
        stats: Pre-computed statistics (optional)
        
    Returns:
        List of suggested questions
    """
    suggester = QuestionSuggester()
    return suggester.generate_suggestions(df, stats)
