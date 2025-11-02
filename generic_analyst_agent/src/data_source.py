"""Data source abstraction and CSV implementation.

- Single Responsibility: provide data access through a simple interface.
- Dependency Inversion: high-level tools depend on BaseDataSource, not concrete pandas CSV reading.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd

try:  # Optional acceleration
    import pyarrow  # type: ignore  # noqa: F401
    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False


class BaseDataSource(ABC):
    """Abstract data source interface.

    Implementations must return a pandas DataFrame when asked for data.
    """

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """Return the dataset as a pandas DataFrame."""
        raise NotImplementedError


class PandasCSVDataSource(BaseDataSource):
    """CSV-backed data source using pandas with large-file optimizations.

    Parameters
    ----------
    filepath : str | Path
        Path to the CSV file to load.
    usecols : Optional[List[str]]
        Subset of columns to load. Reduces IO and memory for large files.
    dtypes : Optional[Dict[str, Any]]
        Explicit pandas dtypes per column. Avoids costly type inference.
    date_cols : Optional[List[str]]
        Column names to parse as datetimes. If None, a heuristic will parse any column containing "date" in its name.
    engine : Optional[str]
        CSV engine: "pyarrow" (if available) or pandas default ("c"). If None, choose "pyarrow" when installed.
    cache_parquet : bool
        When True, materialize a Parquet cache next to the CSV and load from it if fresh; accelerates repeat runs.
    downcast_numeric : bool
        Downcast integer/float columns to save memory.
    categorical_threshold : float
        Convert object columns to category when unique_ratio <= threshold and cardinality <= category_max_cardinality.
    category_max_cardinality : int
        Maximum unique values allowed for category conversion.
    """

    def __init__(
        self,
        filepath: str | Path,
        *,
        usecols: Optional[List[str]] = None,
        dtypes: Optional[Dict[str, Any]] = None,
        date_cols: Optional[List[str]] = None,
        engine: Optional[str] = None,
        cache_parquet: bool = False,
        downcast_numeric: bool = True,
        categorical_threshold: float = 0.2,
        category_max_cardinality: int = 200,
    ) -> None:
        self._path = Path(filepath)
        self._usecols = usecols
        self._dtypes = dtypes
        self._date_cols = date_cols
        self._engine = engine or ("pyarrow" if _HAS_PYARROW else None)
        self._cache_parquet = cache_parquet
        self._downcast_numeric = downcast_numeric
        self._categorical_threshold = categorical_threshold
        self._category_max_cardinality = category_max_cardinality

    def _parquet_path(self) -> Path:
        return self._path.with_suffix(".parquet")

    def _should_use_parquet_cache(self) -> bool:
        pqt = self._parquet_path()
        if not (self._cache_parquet and pqt.exists()):
            return False
        try:
            return pqt.stat().st_mtime >= self._path.stat().st_mtime
        except Exception:
            return False

    def _optimize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Parse date columns
        date_cols = self._date_cols
        if date_cols is None:
            # Heuristic: any column name containing "date"
            date_cols = [c for c in df.columns if "date" in c.lower()]
        for col in date_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass

        # Convert low-cardinality object columns to category
        if self._categorical_threshold > 0:
            for col in df.select_dtypes(include=["object"]).columns:
                try:
                    n = len(df)
                    if n == 0:
                        continue
                    nunique = df[col].nunique(dropna=True)
                    if nunique <= self._category_max_cardinality and (nunique / n) <= self._categorical_threshold:
                        df[col] = df[col].astype("category")
                except Exception:
                    pass

        # Downcast numeric columns
        if self._downcast_numeric:
            for col in df.select_dtypes(include=["int", "int64"]).columns:
                try:
                    df[col] = pd.to_numeric(df[col], downcast="integer")
                except Exception:
                    pass
            for col in df.select_dtypes(include=["float", "float64"]).columns:
                try:
                    df[col] = pd.to_numeric(df[col], downcast="float")
                except Exception:
                    pass
        return df

    def get_data(self) -> pd.DataFrame:
        if not self._path.exists():
            raise FileNotFoundError(f"CSV not found: {self._path}")

        # Prefer Parquet cache for repeat runs
        if self._should_use_parquet_cache():
            try:
                df = pd.read_parquet(self._parquet_path())
                return self._optimize_df(df)
            except Exception:
                # Fall through to CSV if parquet read fails
                pass

        # CSV loading with optional pyarrow engine and column filters
        read_kwargs: Dict[str, Any] = {}
        if self._usecols is not None:
            read_kwargs["usecols"] = self._usecols
        if self._dtypes is not None:
            read_kwargs["dtype"] = self._dtypes
        if self._engine is not None:
            read_kwargs["engine"] = self._engine
        df = pd.read_csv(self._path, **read_kwargs)

        df = self._optimize_df(df)

        # Optionally materialize Parquet cache
        if self._cache_parquet and _HAS_PYARROW:
            try:
                df.to_parquet(self._parquet_path(), index=False)
            except Exception:
                pass
        return df

    def get_data_in_chunks(self, chunksize: int) -> Iterator[pd.DataFrame]:
        """Iterate over the CSV in chunks. Useful for out-of-core workflows.

        Note: The current DataQueryTool expects a single DataFrame and does not
        operate on chunks. This iterator is provided for future extensions.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"CSV not found: {self._path}")
        read_kwargs: Dict[str, Any] = {"chunksize": chunksize}
        if self._usecols is not None:
            read_kwargs["usecols"] = self._usecols
        if self._dtypes is not None:
            read_kwargs["dtype"] = self._dtypes
        if self._engine is not None:
            read_kwargs["engine"] = self._engine
        for chunk in pd.read_csv(self._path, **read_kwargs):
            # Best-effort date parse and optimization per chunk
            yield self._optimize_df(chunk)


class DataFrameDataSource(BaseDataSource):
    """In-memory DataFrame data source.

    Useful for UIs that upload a CSV and hold it in memory, avoiding file round-trips.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def get_data(self) -> pd.DataFrame:
        return self._df
