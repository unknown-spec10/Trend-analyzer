"""Enhanced CSV processor with chunked reading, type inference, and Parquet conversion.

This module implements the three-phase CSV processing strategy:
1. Ingestion: Chunked reading with validation
2. Storage: Parquet conversion and caching
3. Execution: Optimized data access
"""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False

logger = logging.getLogger(__name__)


class OptimizedCSVProcessor:
    """Processes CSV files with memory-efficient chunked reading and Parquet conversion.
    
    Strategy:
    - Phase 1: Read CSV in chunks for memory safety
    - Phase 2: Infer optimal dtypes from sample
    - Phase 3: Convert to Parquet for fast analysis
    """
    
    def __init__(
        self,
        chunk_size: int = 50_000,
        sample_rows: int = 10_000,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize CSV processor.
        
        Args:
            chunk_size: Rows per chunk for initial read (memory safety)
            sample_rows: Rows to sample for type inference
            cache_dir: Directory for Parquet cache (default: temp dir)
        """
        self.chunk_size = chunk_size
        self.sample_rows = sample_rows
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "trend_analyzer_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _infer_optimal_dtypes(self, sample_df: pd.DataFrame) -> Dict[str, str]:
        """Infer optimal dtypes from sample data.
        
        Returns dtype mapping for memory-efficient reading.
        """
        dtypes: Dict[str, str] = {}
        
        for col in sample_df.columns:
            col_data = sample_df[col]
            
            # Skip if already optimal type
            if col_data.dtype in ['int8', 'int16', 'int32', 'float32', 'category', 'datetime64[ns]']:
                dtypes[col] = str(col_data.dtype)
                continue
            
            # Integer optimization
            if pd.api.types.is_integer_dtype(col_data):
                col_min, col_max = col_data.min(), col_data.max()
                if col_min >= -128 and col_max <= 127:
                    dtypes[col] = 'int8'
                elif col_min >= -32768 and col_max <= 32767:
                    dtypes[col] = 'int16'
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    dtypes[col] = 'int32'
                else:
                    dtypes[col] = 'int64'
            
            # Float optimization
            elif pd.api.types.is_float_dtype(col_data):
                # Use float32 for most cases (sufficient precision, half the memory)
                dtypes[col] = 'float32'
            
            # String/Object optimization
            elif pd.api.types.is_object_dtype(col_data):
                nunique = col_data.nunique()
                total = len(col_data.dropna())
                
                if total > 0:
                    uniqueness_ratio = nunique / total
                    
                    # Convert to category if low cardinality (< 20% unique or < 200 unique values)
                    if uniqueness_ratio < 0.2 or nunique < 200:
                        # Note: Don't set as 'category' in dtype dict, convert after reading
                        dtypes[col] = 'object'
                    else:
                        dtypes[col] = 'object'
        
        return dtypes
    
    def _get_cache_path(self, file_hash: str) -> Path:
        """Get Parquet cache file path."""
        return self.cache_dir / f"{file_hash}.parquet"
    
    def _get_stats_cache_path(self, file_hash: str) -> Path:
        """Get statistics cache file path."""
        return self.cache_dir / f"{file_hash}_stats.json"
    
    def _compute_file_hash(self, file_obj: Any) -> str:
        """Compute hash of file for cache key."""
        import hashlib
        
        # For file-like objects
        if hasattr(file_obj, 'read'):
            current_pos = file_obj.tell()
            file_obj.seek(0)
            content = file_obj.read(8192)  # Read first 8KB for hash
            file_obj.seek(current_pos)
            if isinstance(content, str):
                content = content.encode()
            return hashlib.md5(content).hexdigest()
        
        # For file paths
        elif isinstance(file_obj, (str, Path)):
            path = Path(file_obj)
            # Use file size and mtime as hash
            stat = path.stat()
            hash_str = f"{path.name}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(hash_str.encode()).hexdigest()
        
        return hashlib.md5(str(file_obj).encode()).hexdigest()
    
    def process_csv(
        self,
        file_obj: Any,
        force_reload: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process CSV with optimized pipeline.
        
        Args:
            file_obj: File path, file-like object, or bytes
            force_reload: Skip cache and reprocess
            
        Returns:
            Tuple of (DataFrame, statistics_dict)
        """
        file_hash = self._compute_file_hash(file_obj)
        parquet_path = self._get_cache_path(file_hash)
        stats_path = self._get_stats_cache_path(file_hash)
        
        # Try to load from cache
        if not force_reload and parquet_path.exists():
            try:
                logger.info(f"Loading from Parquet cache: {parquet_path}")
                df = pd.read_parquet(parquet_path)
                
                # Load pre-computed statistics
                import json
                stats = {}
                if stats_path.exists():
                    with open(stats_path, 'r') as f:
                        stats = json.load(f)
                
                logger.info(f"Loaded {len(df):,} rows from cache")
                return df, stats
            except Exception as e:
                logger.warning(f"Cache read failed: {e}. Reprocessing CSV.")
        
        # Phase 1: Read sample for type inference
        logger.info(f"Reading first {self.sample_rows} rows for type inference...")
        
        # Handle different input types
        if isinstance(file_obj, (str, Path)):
            sample_df = pd.read_csv(file_obj, nrows=self.sample_rows)
        elif isinstance(file_obj, bytes):
            sample_df = pd.read_csv(io.BytesIO(file_obj), nrows=self.sample_rows)
        else:
            # File-like object
            current_pos = file_obj.tell() if hasattr(file_obj, 'tell') else 0
            sample_df = pd.read_csv(file_obj, nrows=self.sample_rows)
            if hasattr(file_obj, 'seek'):
                file_obj.seek(current_pos)
        
        # Phase 2: Infer optimal dtypes
        logger.info("Inferring optimal data types...")
        optimal_dtypes = self._infer_optimal_dtypes(sample_df)
        logger.info(f"Optimized dtypes: {optimal_dtypes}")
        
        # Phase 3: Read full CSV with optimized types
        logger.info("Reading full CSV with optimized types...")
        
        try:
            if isinstance(file_obj, (str, Path)):
                df = pd.read_csv(file_obj, dtype=optimal_dtypes, low_memory=False)  # type: ignore[arg-type]
            elif isinstance(file_obj, bytes):
                df = pd.read_csv(io.BytesIO(file_obj), dtype=optimal_dtypes, low_memory=False)  # type: ignore[arg-type]
            else:
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                df = pd.read_csv(file_obj, dtype=optimal_dtypes, low_memory=False)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"Optimized read failed: {e}. Falling back to default.")
            # Fallback to default reading
            if isinstance(file_obj, (str, Path)):
                df = pd.read_csv(file_obj, low_memory=False)
            elif isinstance(file_obj, bytes):
                df = pd.read_csv(io.BytesIO(file_obj), low_memory=False)
            else:
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                df = pd.read_csv(file_obj, low_memory=False)
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Phase 4: Pre-compute statistics for LLM context
        logger.info("Pre-computing statistics...")
        stats = self._compute_statistics(df)
        
        # Phase 5: Save to Parquet cache
        if _HAS_PYARROW:
            try:
                logger.info(f"Saving to Parquet cache: {parquet_path}")
                df.to_parquet(parquet_path, index=False, compression='snappy')
                
                # Save statistics
                import json
                with open(stats_path, 'w') as f:
                    json.dump(stats, f)
                
                logger.info("Cache saved successfully")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
        else:
            logger.warning("PyArrow not installed. Install with: pip install pyarrow")
        
        return df, stats
    
    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Pre-compute statistics for fast LLM context generation."""
        stats = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'columns': {}
        }
        
        for col in df.columns:
            col_stats = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isna().sum()),
                'null_percentage': float(df[col].isna().sum() / len(df) * 100),
            }
            
            # Numeric column statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    'min': float(df[col].min()) if not df[col].isna().all() else None,
                    'max': float(df[col].max()) if not df[col].isna().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                    'median': float(df[col].median()) if not df[col].isna().all() else None,
                    'std': float(df[col].std()) if not df[col].isna().all() else None,
                })
            
            # Categorical/Object column statistics
            elif pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
                value_counts = df[col].value_counts()
                col_stats.update({
                    'unique_count': int(df[col].nunique()),
                    'top_values': value_counts.head(5).to_dict() if len(value_counts) > 0 else {},
                })
            
            # Datetime column statistics
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_stats.update({
                    'min_date': str(df[col].min()) if not df[col].isna().all() else None,
                    'max_date': str(df[col].max()) if not df[col].isna().all() else None,
                })
            
            stats['columns'][col] = col_stats
        
        return stats
    
    def clear_cache(self):
        """Clear all cached Parquet files."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleared")
