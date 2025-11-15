"""Zillow ZHVI data ingestion for ZIP codes."""

import pandas as pd
import requests
import io
from pathlib import Path
from src.utils.cache import cache_data, load_cached_data


ZILLOW_ZIP_URL = "https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"


def download_zillow_zip_data(force_download=False):
    """Download Zillow ZHVI data for ZIP codes."""
    cache_key = "zillow_zip_zhvi"
    
    # Check cache
    if not force_download:
        cached = load_cached_data(cache_key, use_parquet=True)
        if cached is not None:
            return cached
    
    try:
        response = requests.get(ZILLOW_ZIP_URL, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        
        # Cache it
        cache_data(df, cache_key, use_parquet=True)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to download Zillow data: {e}")


def get_zip_series(df, zip_code):
    """Extract time series for a specific ZIP code."""
    # ZIP codes in Zillow are stored as integers (e.g., 8901, not 08901)
    # Convert input to various formats for matching
    zip_str = str(zip_code).zfill(5)  # "08901"
    
    # Try to convert to integer (removes leading zeros)
    try:
        zip_int = int(zip_code)
        zip_no_zero = str(zip_int)  # "8901"
    except (ValueError, TypeError):
        zip_int = None
        zip_no_zero = str(zip_code).lstrip('0') or '0'  # Remove leading zeros
    
    # Convert RegionName to string for flexible matching
    df_region_str = df["RegionName"].astype(str)
    df_region_int = df["RegionName"]
    
    # Try multiple matching strategies
    zip_row = None
    
    # Try as string with leading zeros
    mask = df_region_str.str.zfill(5) == zip_str
    if mask.any():
        zip_row = df[mask]
    
    # Try as string without leading zeros
    if zip_row is None or zip_row.empty:
        mask = df_region_str == zip_no_zero
        if mask.any():
            zip_row = df[mask]
    
    # Try as integer
    if (zip_row is None or zip_row.empty) and zip_int is not None:
        mask = df_region_int == zip_int
        if mask.any():
            zip_row = df[mask]
    
    # Try direct string match
    if zip_row is None or zip_row.empty:
        mask = df_region_str == zip_str
        if mask.any():
            zip_row = df[mask]
    
    if zip_row is None or zip_row.empty:
        # Provide helpful error message with sample ZIP codes
        sample_zips = df["RegionName"].astype(str).head(10).tolist()
        raise ValueError(
            f"ZIP code {zip_code} not found in Zillow data. "
            f"Sample available ZIP codes: {', '.join(sample_zips)}. "
            f"Note: ZIP codes are stored without leading zeros (e.g., 8901 instead of 08901)."
        )
    
    # Get the row
    row = zip_row.iloc[0]
    
    # Extract date columns (format: YYYY-MM-DD)
    date_cols = [col for col in df.columns if col.startswith("20") and "-" in col]
    
    # Create series
    values = [row[col] for col in date_cols]
    dates = pd.to_datetime(date_cols)
    
    series = pd.Series(values, index=dates, name="ZHVI")
    series = series.dropna().sort_index()
    
    return series

