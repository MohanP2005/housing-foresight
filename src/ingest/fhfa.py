"""FHFA HPI data (simplified - using synthetic for now)."""

import pandas as pd
import numpy as np
from src.utils.cache import cache_data, load_cached_data


def get_fhfa_hpi(force_download=False):
    """Get FHFA HPI data (synthetic for now)."""
    cache_key = "fhfa_hpi"
    
    # Check cache
    if not force_download:
        cached = load_cached_data(cache_key, use_parquet=True)
        if cached is not None:
            return cached
    
    # Create synthetic HPI data
    dates = pd.date_range("2000-01-01", "2024-12-31", freq="M")
    np.random.seed(42)
    hpi = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    hpi = hpi - hpi[0] + 100  # Start at 100
    
    series = pd.Series(hpi, index=dates, name="hpi")
    
    # Cache it
    cache_data(series.to_frame(), cache_key, use_parquet=True)
    
    return series

