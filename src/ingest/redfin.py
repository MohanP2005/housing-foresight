"""Redfin data (simplified - using synthetic for now)."""

import pandas as pd
import numpy as np
from src.utils.cache import cache_data, load_cached_data


def get_redfin_data(force_download=False):
    """Get Redfin inventory data (synthetic for now)."""
    cache_key = "redfin_inventory"
    
    # Check cache
    if not force_download:
        cached = load_cached_data(cache_key, use_parquet=True)
        if cached is not None:
            return cached
    
    # Create synthetic inventory data
    dates = pd.date_range("2000-01-01", "2024-12-31", freq="M")
    np.random.seed(42)
    inventory = 1000 + np.random.randn(len(dates)) * 200
    inventory = np.clip(inventory, 500, 2000)
    
    series = pd.Series(inventory, index=dates, name="inventory")
    
    # Cache it
    cache_data(series.to_frame(), cache_key, use_parquet=True)
    
    return series

