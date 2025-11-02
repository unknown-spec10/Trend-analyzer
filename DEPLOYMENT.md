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
   GEMINI_API_KEY = "your-gemini-key"  # Optional
   ```

3. **Deploy**
   - Click "Deploy!"
   - Wait 2-5 minutes for installation

### Important Notes

#### Backend API Configuration
- The Streamlit app calls an external backend API (default: `http://localhost:8000`)
- **For cloud deployment**, you have two options:

  **Option A: Deploy backend separately**
  - Deploy the FastAPI backend to a service like Railway, Render, or Fly.io
  - Update the "API base URL" in the Streamlit sidebar to your backend URL
  
  **Option B: Run agent locally in Streamlit** (simpler for testing)
  - Uncomment the local agent code in `app.py` if you want the frontend to run the analysis directly
  - Note: This may hit Streamlit Cloud's memory/timeout limits for large datasets

#### Current Setup
- The app is configured to call an external API
- You'll need to deploy the backend separately or modify `app.py` to run the agent locally

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

**API connection errors:**
- Check the API base URL in the sidebar
- Ensure your backend is deployed and accessible
- Test the backend `/health` endpoint first

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit locally
streamlit run app.py

# Run backend locally (in separate terminal)
python -m uvicorn generic_analyst_agent.src.api:app --reload
```

### Files Structure
```
Trend-analyzer/
├── app.py                          # Streamlit frontend
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies (if needed)
├── .streamlit/
│   ├── config.toml                # Streamlit config
│   └── secrets.toml.template      # Secrets template (DO NOT commit actual secrets)
├── generic_analyst_agent/
│   └── src/
│       ├── api.py                 # FastAPI backend
│       ├── config.py              # Config with Streamlit secrets support
│       └── ...                    # Agent logic
└── .gitignore                     # Excludes secrets, venvs, caches
```

### Next Steps After Deployment
1. Test with a small CSV upload
2. Monitor logs in Streamlit Cloud for any errors
3. Adjust memory settings if needed (in advanced settings)
4. Set up the backend API or modify app to run locally

---

**Need help?** Check Streamlit Cloud logs or open an issue in the repository.
