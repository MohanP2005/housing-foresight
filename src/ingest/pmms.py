"""Freddie Mac PMMS mortgage rate data."""

import pandas as pd
import numpy as np
from src.utils.cache import cache_data, load_cached_data


def get_mortgage_rates(force_download=False):
    """Get mortgage rates (synthetic for now, or use FRED API if available)."""
    cache_key = "pmms_rates"
    
    # Check cache
    if not force_download:
        cached = load_cached_data(cache_key, use_parquet=True)
        if cached is not None:
            return cached
    
    # Create synthetic rates (6% average with variation)
    # In production, you'd use FRED API or download from Freddie Mac
    dates = pd.date_range("2000-01-01", "2024-12-31", freq="M")
    np.random.seed(42)
    rates = 6.0 + np.random.randn(len(dates)) * 0.5
    rates = np.clip(rates, 3.0, 8.0)
    
    series = pd.Series(rates, index=dates, name="mortgage_rate")
    
    # Cache it
    cache_data(series.to_frame(), cache_key, use_parquet=True)
    
    return series

