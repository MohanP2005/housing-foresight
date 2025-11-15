"""Streamlit dashboard for Housing Price Foresight."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.zillow import download_zillow_zip_data, get_zip_series
from src.ingest.pmms import get_mortgage_rates
from src.ingest.redfin import get_redfin_data
from src.ingest.fhfa import get_fhfa_hpi
from src.features.build import build_features
from src.models.sarimax import SARIMAXForecaster
from src.models.xgb import XGBoostForecaster


# Page config
st.set_page_config(
    page_title="Housing Price Foresight",
    page_icon="🏠",
    layout="wide",
)

# Title
st.title("🏠 Housing Price Foresight")
st.markdown("Forecast housing prices by ZIP code using machine learning")

# Sidebar
st.sidebar.header("Configuration")

# ZIP code input
zip_code = st.sidebar.text_input("ZIP Code", value="08901", help="Enter a 5-digit ZIP code")

# Model selection
model_type = st.sidebar.selectbox("Model", ["SARIMAX", "XGBoost"])

# Forecast button
if st.sidebar.button("Generate Forecast", type="primary"):
    with st.spinner("Loading data and generating forecast..."):
        try:
            # Load data
            st.info("📥 Loading data sources...")
            
            # Zillow ZHVI
            zhvi_df = download_zillow_zip_data()
            zhvi_series = get_zip_series(zhvi_df, zip_code)
            
            # Other data sources
            mortgage_rates = get_mortgage_rates()
            inventory = get_redfin_data()
            hpi = get_fhfa_hpi()
            
            st.success("✓ Data loaded successfully")
            
            # Build features
            st.info("🔧 Building features...")
            features_df = build_features(
                zhvi_series=zhvi_series,
                mortgage_rate_series=mortgage_rates,
                inventory_series=inventory,
                hpi_series=hpi,
            )
            st.success("✓ Features built")
            
            # Prepare data for modeling
            y = features_df["ZHVI"].dropna()
            
            # Get exogenous variables
            exog_cols = ["mortgage_rate", "inventory"]
            exog = features_df[exog_cols].reindex(y.index) if all(col in features_df.columns for col in exog_cols) else None
            
            # Fit model
            st.info(f"🤖 Training {model_type} model...")
            
            if model_type == "SARIMAX":
                model = SARIMAXForecaster()
                model.fit(y, exog=exog)
                
                # Generate 5-year forecast (60 months)
                # For future exogenous variables, use last values
                if exog is not None:
                    last_rate = exog["mortgage_rate"].iloc[-1]
                    last_inventory = exog["inventory"].iloc[-1]
                    # Use month end frequency to match forecast dates
                    start_date = pd.Timestamp(y.index[-1]).replace(day=1) + pd.DateOffset(months=1)
                    future_exog = pd.DataFrame({
                        "mortgage_rate": [last_rate] * 60,
                        "inventory": [last_inventory] * 60,
                    }, index=pd.date_range(start=start_date, periods=60, freq="ME"))
                else:
                    future_exog = None
                
                forecast = model.predict(60, exog=future_exog)
            else:
                # XGBoost
                feature_cols = [col for col in features_df.columns if col not in ["ZHVI", "month"]]
                X = features_df[feature_cols].reindex(y.index)
                
                model = XGBoostForecaster()
                model.fit(X, y)
                
                # Create future features (use last values)
                future_dates = pd.date_range(start=y.index[-1] + pd.DateOffset(months=1), periods=60, freq="M")
                future_X = pd.DataFrame(index=future_dates)
                
                for col in feature_cols:
                    if col in features_df.columns:
                        last_val = features_df[col].iloc[-1]
                        future_X[col] = last_val
                    else:
                        future_X[col] = 0.0
                
                forecast = model.predict(future_X, start_value=y.iloc[-1])
            
            st.success("✓ Forecast generated")
            
            # Display results
            st.header("📊 Forecast Results")
            
            # Combine historical and forecast
            historical = y.tail(24)  # Last 2 years
            combined = pd.concat([historical, forecast])
            combined.name = "Home Value"
            
            # Create plot
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical.index,
                y=historical.values,
                mode="lines",
                name="Historical",
                line=dict(color="blue", width=2),
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode="lines",
                name="Forecast",
                line=dict(color="red", width=2, dash="dash"),
            ))
            
            fig.update_layout(
                title=f"5-Year Housing Price Forecast for ZIP {zip_code}",
                xaxis_title="Date",
                yaxis_title="Home Value ($)",
                hovermode="x unified",
                template="plotly_white",
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Value", f"${y.iloc[-1]:,.0f}")
            with col2:
                forecast_5yr = forecast.iloc[-1]
                st.metric("5-Year Forecast", f"${forecast_5yr:,.0f}")
            with col3:
                pct_change = ((forecast_5yr / y.iloc[-1]) - 1) * 100
                st.metric("5-Year Change", f"{pct_change:.1f}%")
            
            # Download CSV
            forecast_df = forecast.to_frame(name="Forecast")
            csv = forecast_df.to_csv()
            st.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name=f"forecast_{zip_code}.csv",
                mime="text/csv",
            )
            
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.info("💡 Try a different ZIP code. Not all ZIP codes are available in the Zillow dataset.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Please check your inputs and try again.")

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Enter a 5-digit ZIP code
2. Select a model (SARIMAX or XGBoost)
3. Click "Generate Forecast"
4. View the 5-year forecast chart
5. Download results as CSV
""")

