"""Configuration for CSV processing and sandboxed execution.

Centralized settings for memory limits, timeouts, and optimization thresholds.
"""
from pathlib import Path
import tempfile

# ==================== CSV Processing ====================

# Chunked reading settings
CSV_CHUNK_SIZE = 50_000  # Rows per chunk for initial read
CSV_SAMPLE_ROWS = 10_000  # Rows to sample for type inference

# Type optimization thresholds
CATEGORICAL_THRESHOLD = 0.2  # Convert to category if < 20% unique
CATEGORY_MAX_CARDINALITY = 200  # Max unique values for category

# Cache settings
CACHE_DIR = Path(tempfile.gettempdir()) / "trend_analyzer_cache"
ENABLE_PARQUET_CACHE = True  # Enable Parquet caching for repeat runs
CACHE_COMPRESSION = 'snappy'  # Parquet compression: 'snappy', 'gzip', 'brotli'

# ==================== Sandboxed Execution ====================

# Process isolation
USE_SUBPROCESS_SANDBOX = True  # Use subprocess for code execution (RECOMMENDED)
EXECUTION_TIMEOUT_SECONDS = 30  # Max execution time per query
MEMORY_LIMIT_MB = 512  # Max memory per subprocess (Unix only)

# Code safety
ALLOW_IMPORTS = False  # Whether to allow import statements in generated code
MAX_RETRIES = 2  # Number of retry attempts for failed code generation

# ==================== Data Limits ====================

# File size limits
MAX_CSV_SIZE_MB = 500  # Maximum CSV file size to process
MAX_ROWS_IN_MEMORY = 1_000_000  # Switch to Dask above this threshold
MAX_COLUMNS = 500  # Maximum number of columns

# Preview settings
PREVIEW_ROWS = 10  # Rows to show in data preview
STATS_TOP_VALUES = 5  # Number of top values to show in statistics

# ==================== Performance ====================

# Parallel processing
USE_MULTICORE = True  # Enable multicore processing where applicable
N_CORES = None  # Number of cores (None = auto-detect)

# Memory optimization
DOWNCAST_NUMERIC = True  # Downcast int64/float64 to smaller types
AUTO_CATEGORIZE = True  # Auto-convert low-cardinality strings to category

# ==================== Logging ====================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_CODE_GENERATION = True  # Log generated pandas code
LOG_EXECUTION_TIME = True  # Log execution timing

# ==================== Export Settings ====================

def get_cache_dir() -> Path:
    """Get cache directory, creating if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def get_config_summary() -> dict:
    """Get configuration summary for display."""
    return {
        "CSV Processing": {
            "Chunk Size": f"{CSV_CHUNK_SIZE:,} rows",
            "Sample Rows": f"{CSV_SAMPLE_ROWS:,} rows",
            "Parquet Cache": "Enabled" if ENABLE_PARQUET_CACHE else "Disabled",
            "Cache Directory": str(CACHE_DIR),
        },
        "Execution Safety": {
            "Subprocess Isolation": "Enabled" if USE_SUBPROCESS_SANDBOX else "Disabled",
            "Timeout": f"{EXECUTION_TIMEOUT_SECONDS}s",
            "Memory Limit": f"{MEMORY_LIMIT_MB}MB" if MEMORY_LIMIT_MB else "Unlimited",
            "Max Retries": MAX_RETRIES,
        },
        "Data Limits": {
            "Max File Size": f"{MAX_CSV_SIZE_MB}MB",
            "Max Rows": f"{MAX_ROWS_IN_MEMORY:,}",
            "Max Columns": MAX_COLUMNS,
        },
        "Optimizations": {
            "Downcast Numeric": "Enabled" if DOWNCAST_NUMERIC else "Disabled",
            "Auto Categorize": "Enabled" if AUTO_CATEGORIZE else "Disabled",
            "Multicore": "Enabled" if USE_MULTICORE else "Disabled",
        }
    }
