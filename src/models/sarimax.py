"""SARIMAX model for forecasting."""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")


class SARIMAXForecaster:
    """SARIMAX forecasting model."""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.last_date = None  # Store last date from training data
    
    def fit(self, y, exog=None):
        """Fit the model."""
        # Store the last date from training data
        self.last_date = y.index[-1]
        
        try:
            self.model = SARIMAX(y, exog=exog, order=self.order, seasonal_order=self.seasonal_order)
            self.fitted_model = self.model.fit(disp=False, maxiter=200)
        except Exception:
            # Fallback to simpler model
            self.model = SARIMAX(y, exog=exog, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
            self.fitted_model = self.model.fit(disp=False, maxiter=200)
        return self
    
    def predict(self, steps, exog=None):
        """Predict future values."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        if self.last_date is None:
            raise ValueError("Model must be fitted first (last_date not set)")
        
        # Use get_forecast for better results with confidence intervals
        try:
            forecast_result = self.fitted_model.get_forecast(steps=steps, exog=exog)
            forecast = forecast_result.predicted_mean
        except Exception:
            # Fallback to simple forecast
            forecast = self.fitted_model.forecast(steps=steps, exog=exog)
        
        # Create proper monthly date index starting from the month after last_date
        # Simple approach: start from next month using month end frequency
        start_date = pd.Timestamp(self.last_date).replace(day=1) + pd.DateOffset(months=1)
        dates = pd.date_range(start=start_date, periods=steps, freq="ME")  # ME = Month End
        
        # Ensure forecast is not NaN
        if forecast.isna().all():
            # If all NaN, use last value as fallback
            if hasattr(self.fitted_model, 'fittedvalues') and len(self.fitted_model.fittedvalues) > 0:
                last_value = self.fitted_model.fittedvalues.iloc[-1]
                forecast = pd.Series([last_value] * steps, index=dates)
            else:
                raise ValueError("Forecast returned all NaN values")
        
        return pd.Series(forecast.values, index=dates, name="Forecast")

