"""XGBoost model for forecasting."""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


class XGBoostForecaster:
    """XGBoost forecasting model."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.last_value = None
    
    def fit(self, X, y):
        """Fit the model on percentage changes."""
        self.last_value = y.iloc[-1]
        
        # Create target: percentage changes
        y_pct = y.pct_change().dropna() * 100
        
        # Align X and y
        aligned_idx = y_pct.index.intersection(X.index)
        X_aligned = X.loc[aligned_idx]
        y_pct_aligned = y_pct.loc[aligned_idx]
        
        # Remove non-numeric columns
        numeric_cols = X_aligned.select_dtypes(include=[np.number]).columns.tolist()
        X_features = X_aligned[numeric_cols]
        
        # Remove NaN
        valid_mask = ~(X_features.isna().any(axis=1) | y_pct_aligned.isna())
        X_features = X_features.loc[valid_mask]
        y_pct_aligned = y_pct_aligned.loc[valid_mask]
        
        self.feature_columns = X_features.columns.tolist()
        
        # Scale
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_features),
            index=X_features.index,
            columns=X_features.columns,
        )
        
        # Fit model
        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
        self.model.fit(X_scaled, y_pct_aligned)
        
        return self
    
    def predict(self, X, start_value=None):
        """Predict by reconstructing level from percentage changes."""
        if start_value is None:
            start_value = self.last_value
        
        # Prepare features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_features = X[numeric_cols]
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in X_features.columns:
                X_features[col] = 0.0
        
        X_features = X_features[self.feature_columns]
        X_features = X_features.fillna(0)
        
        # Scale
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_features),
            index=X_features.index,
            columns=X_features.columns,
        )
        
        # Predict percentage changes
        pct_changes = self.model.predict(X_scaled)
        
        # Reconstruct level
        predictions = [start_value]
        current = start_value
        for pct in pct_changes:
            current = current * (1 + pct / 100)
            predictions.append(current)
        
        return pd.Series(predictions[1:], index=X.index)

