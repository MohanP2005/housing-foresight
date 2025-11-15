"""Local caching utilities."""

import pandas as pd
from pathlib import Path
import pickle
import hashlib


CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_key(*args, **kwargs):
    """Generate cache key from arguments."""
    key_str = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_str.encode()).hexdigest()


def cache_data(data, key, use_parquet=True):
    """Cache data to disk."""
    cache_path = CACHE_DIR / f"{key}.parquet" if use_parquet else CACHE_DIR / f"{key}.pkl"
    
    if use_parquet and isinstance(data, pd.DataFrame):
        data.to_parquet(cache_path, engine="pyarrow", compression="snappy")
    else:
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
    
    return cache_path


def load_cached_data(key, use_parquet=True):
    """Load cached data from disk."""
    cache_path = CACHE_DIR / f"{key}.parquet" if use_parquet else CACHE_DIR / f"{key}.pkl"
    
    if not cache_path.exists():
        return None
    
    try:
        if use_parquet:
            return pd.read_parquet(cache_path, engine="pyarrow")
        else:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
    except Exception:
        return None

