# Trend Analyzer - Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

### Prerequisites
1. Push your code to GitHub (repository: `Trend-analyzer`)
2. Sign up at [share.streamlit.io](https://share.streamlit.io)

### Deployment Steps

1. **Connect Repository**
   - Click "New app" in Streamlit Cloud
   - Select your GitHub repository: `unknown-spec10/Trend-analyzer`
   - Set main file path: `app.py`
   - Branch: `main`

2. **Configure Secrets**
   - Go to app settings → Secrets
   - Add your API keys in TOML format:
   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   GOOGLE_API_KEY = "your-google-api-key"
   GOOGLE_CSE_ID = "your-cse-id"
   GEMINI_API_KEY = "your-gemini-key"  # Required for fallback and question generation
   ```

3. **Deploy**
   - Click "Deploy!"
   - Wait 2-5 minutes for installation

### Important Notes

#### Current Architecture
- **Self-Contained**: The Streamlit app runs the agent locally (no separate backend needed)
- **Dual LLM System**: Automatically falls back to Gemini if Groq hits rate limits
- **Smart Validation**: Pre-validates questions before LLM execution to save API calls
- **Session Caching**: Results cached for 1 hour to reduce redundant API calls
- **Memory Optimized**: Uses Parquet caching and type downcasting for efficient data handling

#### Performance Considerations
- **Streamlit Cloud Limits**:
  - 1GB RAM per app
  - CPU limits may affect large dataset processing
  - Consider upgrading to Streamlit Cloud Team/Enterprise for production use
- **Optimization Features**:
  - File size limit: 500MB (configurable)
  - Row limit: 1M rows (configurable)
  - Chunked CSV reading: 50K rows per chunk
  - Memory reduction: ~75% via type optimization

### Troubleshooting

**Build failures with pandas/pyarrow:**
- Fixed by using binary wheels (pandas 2.2.3)
- Removed pyarrow from requirements (pandas includes it)

**Missing dependencies:**
- All required packages are in `requirements.txt`
- System packages can be added to `packages.txt` if needed

**Secrets not loading:**
- Check that secrets are in TOML format in Streamlit Cloud UI
- Verify secret keys match exactly: `GROQ_API_KEY`, `GOOGLE_API_KEY`, etc.

**LLM Rate Limit Errors:**
- System should automatically use Gemini fallback
- Check logs for "Rate limit detected, using Gemini fallback"
- Verify `GEMINI_API_KEY` is properly configured in secrets
- Monitor Groq free tier limits: 14,400 requests/day

**Question Validation Blocking:**
- If legitimate questions are blocked, check dataset column names
- System uses substring matching (e.g., "temperature" matches "temperature_celsius")
- Temporal questions require date columns in the dataset

### Local Development

```powershell
# Create virtual environment
python -m venv myenv
.\myenv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Create .env file with API keys
# (see generic_analyst_agent/.env.template)

# Run Streamlit locally
streamlit run app.py

# Run tests
python tests/test_query.py "data/test_employees.csv" "What is the average salary?"
```

### Files Structure
```
Trend-analyzer/
├── app.py                              # Streamlit UI (main entry point)
├── requirements.txt                    # Python dependencies
├── packages.txt                        # System dependencies (if needed)
├── data/                               # Sample datasets
│   ├── insurance_data.csv
│   ├── test_employees.csv
│   ├── test_sales.csv
│   └── test_temperature.csv
├── tests/                              # Test suite
│   ├── test_query.py                   # Test pandas code generation
│   ├── test_conversation.py            # Test conversation history
│   ├── test_suggestions.py             # Test question generation
│   └── test_cache.py                   # Test session caching
├── generic_analyst_agent/
│   ├── .env                            # API keys (local only, gitignored)
│   └── src/
│       ├── agent_graph.py              # LangGraph orchestration
│       ├── tools.py                    # DataQueryTool with Gemini fallback
│       ├── prompts.py                  # Dynamic adaptive prompts
│       ├── config.py                   # Config loader (supports Streamlit secrets)
│       ├── data_source.py              # Data source abstraction
│       ├── csv_processor.py            # Optimized CSV processing
│       ├── sandbox.py                  # Sandboxed code execution
│       ├── question_suggester.py       # AI question generation (Gemini)
│       └── session_cache.py            # Result caching
└── .gitignore                          # Excludes secrets, venvs, caches
```

### Next Steps After Deployment
1. **Test Basic Functionality**:
   - Upload a small CSV (< 10MB)
   - Ask a simple aggregation question: "What is the average value?"
   - Verify answer appears correctly

2. **Test Multi-Metric Questions**:
   - Ask: "What are the minimum, maximum, and average of [column] by [category]?"
   - Verify all three metrics appear in the answer

3. **Test Validation**:
   - Ask about non-existent columns
   - Verify error message with helpful suggestions

4. **Monitor Performance**:
   - Check Streamlit Cloud logs for any warnings
   - Watch for rate limit messages and Gemini fallback triggers
   - Adjust memory settings if needed (in advanced settings)

5. **Test Question Suggestions**:
   - Verify 5-7 AI-generated questions appear in sidebar
   - Click suggestions to auto-fill questions

6. **Production Readiness**:
   - Consider upgrading Streamlit Cloud tier for production use
   - Monitor API usage (Groq, Gemini, Google CSE quotas)
   - Set up alerts for failures or rate limits

---

**Need help?** Check Streamlit Cloud logs or open an issue in the repository.
