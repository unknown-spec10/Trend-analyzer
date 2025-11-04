# Trend Analyzer

A dynamic, dataset-agnostic data analysis agent that combines internal data insights with external web research to answer complex analytical questions. Built with LangGraph, LangChain, and Streamlit.

## 🌟 Key Features

### Intelligent Adaptive Workflow
- **Smart Decision Making**: Automatically determines if external web research is needed
- **Dynamic Analysis**: Works with ANY CSV structure without hardcoded assumptions
- **Multi-dimensional Queries**: Handles complex questions involving multiple data dimensions
- **Source Citations**: Provides clickable links to all external sources used in analysis

### Architecture Highlights
- **Dataset Agnostic**: Inspects CSV schema dynamically and adapts analysis approach
- **Conditional Routing**: LLM judges if internal data is sufficient before searching web
- **Context-Aware Search**: Generates targeted search queries based on actual data patterns
- **Structured Output**: Returns standardized JSON with metrics, values, segments, and details

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

### Example Questions

**Medical Insurance Dataset:**
- "Which region has claimed most and which gender in that region has claimed more?"
- "What are the possible causes for high claims in the southeast region?"

**Sales Dataset:**
- "Which product category had the highest revenue last quarter?"
- "Why did sales drop in the northeast region?"

**Any CSV:**
- "What patterns do you see in the data?"
- "What are the top 3 segments by volume?"

## 🏗️ Project Structure

```
Trend-analyzer/
├── app.py                              # Streamlit web UI
├── test_agent.py                       # CLI test script
├── generic_analyst_agent/
│   ├── .env                            # API keys (create this)
│   ├── src/
│   │   ├── config.py                   # Configuration loader
│   │   ├── data_source.py              # Data source abstraction
│   │   ├── tools.py                    # Data query & search tools
│   │   ├── agent_graph.py              # LangGraph orchestration
│   │   └── prompts.py                  # System prompts
│   └── README.md                       # Package documentation
├── requirements.txt
└── README.md                           # This file
```

## 🔧 Technical Details

### Technologies
- **LangGraph**: Agent orchestration with conditional workflows
- **LangChain**: Tool integration and LLM abstraction
- **Groq (Llama 3.1 8B)**: Fast LLM inference for code generation
- **Google Custom Search**: Web research capabilities
- **Google Gemini 2.5 Flash**: Web content synthesis
- **Streamlit**: Interactive web UI
- **Pandas**: Data manipulation

### Code Execution Safety
The `DataQueryTool` uses Python's `exec()` to run LLM-generated pandas code. Current safety measures:
- Restricted `__builtins__` (limited to safe functions)
- No import statements allowed
- Code sanitization and validation
- Captured stdout for controlled output

**⚠️ Production Warning**: For production use, implement:
- Process isolation (containers, VMs)
- Resource limits (CPU, memory, time)
- Stricter sandboxing
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
- Provides column-specific guidance to LLM
- Includes example code for multi-dimensional queries
- Adjusts based on dataset structure

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
- **Chat Interface**: Conversational Q&A about your data
- **Context Toggle**: Show/hide internal facts and external research
- **Source Citations**: Clickable links to all web sources
- **Conversation History**: All Q&A pairs preserved in session

## 🧪 Testing

### Test with Sample Data

```powershell
# Download or create a CSV with any structure
# Run test script
python test_agent.py
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
- Groq: 14,400 requests/day, 6,000 tokens/minute
- Google CSE: 100 queries/day (free tier)
- Consider upgrading for production use

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
