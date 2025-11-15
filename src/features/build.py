"""Build features from time series data."""

import pandas as pd
import numpy as np


def build_features(zhvi_series, mortgage_rate_series, inventory_series, hpi_series):
    """Build feature DataFrame from time series."""
    # Align all series to common index
    all_series = {
        "ZHVI": zhvi_series,
        "mortgage_rate": mortgage_rate_series,
        "inventory": inventory_series,
        "hpi": hpi_series,
    }
    
    # Create common date range
    all_dates = set()
    for series in all_series.values():
        if series is not None:
            all_dates.update(series.index)
    
    common_index = pd.date_range(min(all_dates), max(all_dates), freq="M")
    
    # Build DataFrame
    df = pd.DataFrame(index=common_index)
    
    for name, series in all_series.items():
        if series is not None:
            df[name] = series.reindex(common_index).ffill().bfill()
    
    # Create lags
    for col in ["ZHVI", "mortgage_rate", "inventory"]:
        if col in df.columns:
            df[f"{col}_lag_1"] = df[col].shift(1)
            df[f"{col}_lag_12"] = df[col].shift(12)
    
    # Create percentage changes
    if "ZHVI" in df.columns:
        df["ZHVI_pct"] = df["ZHVI"].pct_change() * 100
    
    # Seasonality
    df["month"] = df.index.month
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
    df = pd.concat([df, month_dummies], axis=1)
    
    # Remove rows with missing target
    df = df.dropna(subset=["ZHVI"])
    
    return df

