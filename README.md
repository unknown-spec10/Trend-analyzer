# Trend Analyzer

A dynamic, dataset-agnostic data analysis agent that combines internal data insights with external web research to answer complex analytical questions. Built with LangGraph, LangChain, and Streamlit.

## 🌟 Key Features

### Intelligent Adaptive Workflow
- **Smart Decision Making**: Automatically determines if external web research is needed
- **Dynamic Analysis**: Works with ANY CSV structure without hardcoded assumptions
- **Multi-Metric Support**: Handles questions requesting multiple statistics (min, max, average, etc.)
- **Intelligent Validation**: Pre-validates questions against dataset columns to prevent errors
- **Dual LLM System**: Groq (primary) with automatic Gemini fallback on rate limits
- **Source Citations**: Provides clickable links to all external sources used in analysis
- **Conversation History**: Follow-up questions automatically reference previous context
- **Question Suggestions**: AI generates 5-7 relevant questions based on your dataset
- **Session Caching**: Repeated questions use cached results (1-hour TTL)
- **Clean Formatting**: Automatically formats summaries with proper bullet points and number formatting

### Architecture Highlights
- **Dataset Agnostic**: Inspects CSV schema dynamically and adapts analysis approach
- **Conditional Routing**: LLM judges if internal data is sufficient before searching web
- **Context-Aware Search**: Generates targeted search queries based on actual data patterns
- **Structured Output**: Returns standardized JSON with metrics, values, segments, and details
- **Conversation Context**: Detects ambiguous follow-ups like "What about females?" and expands them
- **Performance Optimization**: Parquet caching, type downcasting, session result caching

## 🚀 Quickstart

### 1. Setup Environment

```powershell
# Clone the repository
git clone https://github.com/unknown-spec10/Trend-analyzer.git
cd Trend-analyzer

# Create and activate virtual environment
python -m venv myenv
.\myenv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in `generic_analyst_agent/` directory:

```env
# Required
GROQ_API_KEY=gsk_...              # Groq API for LLM inference
GOOGLE_API_KEY=AIza...             # Google Cloud API key
GOOGLE_CSE_ID=...                  # Custom Search Engine ID
GEMINI_API_KEY=AIza...             # Gemini API for web synthesis

# Optional (for Streamlit Cloud deployment)
# Configure these in Streamlit Cloud secrets instead
```

#### Getting API Keys:

1. **Groq API Key**: Sign up at [console.groq.com](https://console.groq.com)
2. **Google Custom Search**:
   - Create project at [console.cloud.google.com](https://console.cloud.google.com)
   - Enable Custom Search JSON API
   - Create API key and Custom Search Engine at [cse.google.com](https://cse.google.com)
3. **Gemini API**: Get key at [ai.google.dev](https://ai.google.dev)

### 3. Run the Application

#### Streamlit Web UI (Recommended)

```powershell
streamlit run app.py
```

Navigate to `http://localhost:8501`, upload your CSV, and start asking questions!

#### Command Line Test

```powershell
python test_agent.py
```

## 📊 How It Works

### Agent Workflow

```
START → Analyze Internal Data → Decide if Search Needed → [Conditional]
                                                              ├─→ Search Web (if needed)
                                                              └─→ Synthesize Answer → END
```

1. **Data Analysis**: LLM generates pandas code to query your CSV
2. **Smart Decision**: LLM evaluates if internal data fully answers the question
3. **Web Research** (conditional): Searches for external factors and causes
4. **Synthesis**: Combines data insights with research into comprehensive answer


## 🏗️ Project Structure

```
Trend-analyzer/
├── app.py                              # Streamlit web UI
├── tests/                              # Test files (in .gitignore)
│   ├── test_agent.py                   # End-to-end workflow test
│   ├── test_conversation.py            # Conversation history test
│   ├── test_suggestions.py             # Question suggester test
│   ├── test_cache.py                   # Session caching test
│   └── test_csv_processing.py          # CSV optimization test
├── generic_analyst_agent/
│   ├── .env                            # API keys (create this)
│   ├── src/
│   │   ├── config.py                   # Configuration loader
│   │   ├── data_source.py              # Data source abstraction
│   │   ├── tools.py                    # Data query & search tools
│   │   ├── agent_graph.py              # LangGraph orchestration
│   │   ├── prompts.py                  # System prompts
│   │   ├── csv_processor.py            # Optimized CSV processing
│   │   ├── sandbox.py                  # Sandboxed code execution
│   │   ├── processing_config.py        # Configuration management
│   │   ├── question_suggester.py       # AI question generation
│   │   └── session_cache.py            # Result caching
│   └── README.md                       # Package documentation
├── requirements.txt
└── README.md                           # This file
```

## 🔧 Technical Details

### Technologies
- **LangGraph**: Agent orchestration with conditional workflows
- **LangChain**: Tool integration and LLM abstraction
- **Groq (Llama 3.1 8B)**: Primary LLM for fast code generation (temperature=0.2)
- **Google Gemini 2.0 Flash Exp**: Fallback LLM for rate limits and question generation
- **Google Custom Search**: Web research capabilities
- **Streamlit**: Interactive web UI with live agent status updates
- **Pandas**: Data manipulation and analysis

### Code Execution Safety
The `DataQueryTool` uses sandboxed execution for LLM-generated pandas code. Current safety measures:
- **Question Validation**: Pre-checks if question matches dataset columns
- **Subprocess Isolation**: Code runs in separate process (Unix) with resource limits
- **Memory Limit**: 512MB cap on code execution
- **Time Limit**: 30-second timeout for queries
- **Restricted Builtins**: No file I/O, no imports, limited functions
- **Code Sanitization**: Removes dangerous patterns before execution
- **Error Recovery**: Automatic Gemini fallback on code generation failures

**⚠️ Production Warning**: For high-security production use, implement:
- Container isolation (Docker, Kubernetes)
- Network restrictions
- Audit logging
- Code review/approval workflows

### Dynamic Features

#### 1. Column Type Detection
```python
# Automatically detects:
- Numeric columns: [age, bmi, claim, ...]
- Categorical columns: [gender, region, smoker, ...]
- Date columns: [date, timestamp, ...]
```

#### 2. Adaptive Prompts
- **Fully Dynamic**: No hardcoded column names or assumptions
- **Column-Specific Guidance**: Tailored instructions based on actual columns
- **Multi-Metric Examples**: Shows how to calculate min, max, mean simultaneously
- **Format Guidelines**: Instructs LLM to use clean bullet points and number formatting
- **Context-Aware**: Adjusts based on dataset structure (numeric, categorical, date columns)

#### 3. Intelligent Search
- LLM generates contextual search queries
- Filters results based on relevance scoring
- Iterates up to 3 times for best matches

#### 4. Structured Output
```json
{
  "metric": "regional_claim_analysis",
  "value": 5784925,
  "period": "full_dataset",
  "segment": "southeast",
  "unit": "USD",
  "details": {
    "by_gender": {"male": 3188798, "female": 2596127},
    "top_gender": "male"
  },
  "summary": "Southeast region has highest claims..."
}
```

## 📱 Streamlit UI Features

- **CSV Upload**: Drag-and-drop or browse for any CSV file
- **Data Preview**: See first 10 rows and column info
- **Quick Statistics**: Memory usage, null counts, row/column totals
- **Chat Interface**: Modern chat input with auto-clear after submission
- **Live Agent Status**: Expandable status box showing agent thinking steps
  - 📊 Analyzing internal data...
  - 🤔 Determining if external research is needed...
  - 🧠 Processing your question...
  - ✅ Analysis complete!
- **Context Display**: Toggle to show/hide:
  - 📊 Dataset Context (columns, types, sample data)
  - 🐼 Pandas Query (concise one-line query used)
  - 🧪 Full Generated Code (expandable)
- **Clean Formatting**: Properly formatted summaries with bullet points and currency
- **Source Citations**: Clickable links to all web sources
- **Conversation History**: All Q&A pairs preserved in session
- **Question Suggestions**: AI-generated questions in sidebar (5-7 suggestions)
  - Gemini-based generation with ultimate fallback
  - Click any suggestion to auto-fill the input
  - Based on dataset schema and statistics
  - Mix of aggregation, comparison, trend, and outlier questions
- **Session Caching**: Repeated questions return cached results instantly
  - 1-hour TTL (time-to-live)
  - Case-insensitive matching
  - Dataset-aware (cache invalidated on new upload)

## 💬 Conversation Features

### Multi-Turn Conversations
The agent maintains conversation context across multiple questions:

**Example:**
```
User: "Which region has claimed the most?"
Agent: "Southeast region has highest claims at $5.78M..."

User: "What about females?"  ← Ambiguous follow-up
Agent: [Detects context reference]
      [Expands to: "What is the total claimed by females in the region that claimed the most?"]
      "Females in the southeast region claimed $2.60M..."
```

### Context Detection Patterns
The agent automatically detects follow-up questions containing:
- Pronouns: "that", "this", "it", "those", "these"
- Follow-up phrases: "what about", "how about", "and for"
- Comparison words: "same", "similar", "different"
- Single-word references: "females?", "males?", "other"

### Question Expansion
When a follow-up is detected:
1. Retrieves previous question + answer from history
2. Uses LLM to expand the follow-up into standalone question
3. Preserves original intent while adding necessary context
4. Executes analysis on the expanded question

## ⚡ Performance Optimization

### CSV Processing Pipeline (3-Phase Strategy)

#### Phase 1: Ingestion
- **Chunked Reading**: Processes large files in 50,000-row chunks
- **Memory Management**: Avoids loading entire dataset into memory at once
- **Type Inference**: Samples 10,000 rows to determine optimal data types
- **Size Limits**: 500MB file size, 1M row limits (configurable)

#### Phase 2: Storage & Transformation
- **Type Optimization**: Downcasts int64→int16/int32, float64→float32 where safe
- **Parquet Caching**: Stores optimized version for 0.1x-100x faster reloads
- **Pre-computed Statistics**: Column stats, null counts, memory usage
- **Cache Invalidation**: SHA-256 hash-based detection of data changes

#### Phase 3: Execution Optimization
- **Sandboxed Execution**: Subprocess isolation with resource limits
  - Memory limit: 512MB
  - Time limit: 30s
  - Restricted builtins (no file I/O, no imports)
- **Statistics-Enriched Prompts**: LLM receives pre-computed stats for faster queries
- **Session Caching**: Duplicate questions return cached results (1-hour TTL)

### Performance Metrics (Insurance Dataset: 1340 rows, 11 columns)
- **Memory**: 0.3MB (optimized) vs 1.2MB (raw) = 75% reduction
- **Reload Time**: Parquet cache = 10ms vs CSV = 100ms = 10x faster
- **Cached Query**: <1ms vs Fresh Query = ~5-10s = >1000x faster

## 🧪 Testing

### Run Test Suite

```powershell
# Test conversation history
python tests/test_conversation.py

# Test question suggester
python tests/test_suggestions.py

# Test session caching
python tests/test_cache.py

# Test CSV processing
python tests/test_csv_processing.py

# Test end-to-end workflow
python tests/test_agent.py
```

### Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

View generated pandas code and execution results in console.

## 🎯 Design Principles

### SOLID Architecture
- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Extensible without modifying core logic
- **Dependency Inversion**: Abstractions (BaseDataSource) not implementations

### No Hardcoded Assumptions
- ❌ No domain-specific fallbacks (medical, retail, etc.)
- ✅ Generic dataset overview on failure
- ✅ Dynamic column detection
- ✅ Adaptive analysis strategies

### Retry Logic with Feedback
- LLM code generation attempts: 2 tries
- Error messages fed back to LLM for correction
- Graceful fallback to dataset statistics

## 🐛 Troubleshooting

### API Errors

**403 Forbidden (Google Custom Search)**:
```
1. Visit https://console.developers.google.com/apis/library/customsearch.googleapis.com
2. Select your project
3. Click "Enable"
4. Wait a few minutes and retry
```

**Rate Limits**:
- **Groq**: 14,400 requests/day, 6,000 tokens/minute
  - System automatically switches to Gemini fallback on rate limits
  - Check logs for "Rate limit detected, using Gemini fallback"
- **Google Gemini**: 1,500 requests/day (free tier)
- **Google CSE**: 100 queries/day (free tier)
- Consider upgrading for production use

**Gemini Fallback Not Working**:
- Verify `GEMINI_API_KEY` is set in `.env`
- Check logs for fallback trigger messages
- Ensure google-generativeai package is installed

### Import Errors

```powershell
# Ensure venv is activated
.\myenv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Dataset Issues

**CSV not loading**:
- Check file encoding (UTF-8 recommended)
- Verify CSV format (comma-separated, headers in first row)
- Try smaller file first (< 10MB)

**Incorrect answers**:
- Check DEBUG logs to see generated pandas code
- Verify column names match question terms
- Simplify question or add more context

## 📈 Roadmap

- [ ] Support for multiple file formats (Excel, JSON, Parquet)
- [ ] Advanced visualizations (charts, graphs)
- [ ] Query history and bookmarks
- [ ] Export results to PDF/Word
- [ ] Multi-turn conversations with context
- [ ] Custom data transformations

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/unknown-spec10/Trend-analyzer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/unknown-spec10/Trend-analyzer/discussions)

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Powered by [Groq](https://groq.com) and [Google AI](https://ai.google.dev)
- UI framework: [Streamlit](https://streamlit.io)

---

**Made with ❤️ for data analysts everywhere**
