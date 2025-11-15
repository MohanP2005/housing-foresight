# 🏠 Housing Price Foresight

A machine learning-powered web application for forecasting housing prices by ZIP code using SARIMAX and XGBoost models.

## Features

- **5-Year Housing Price Forecasts**: Predict future home values for any US ZIP code
- **Multiple ML Models**: Choose between SARIMAX (time series) or XGBoost (gradient boosting)
- **Interactive Dashboard**: Beautiful Streamlit interface with interactive charts
- **Real-time Data**: Automatically fetches data from Zillow, FRED, Redfin, and FHFA
- **Export Results**: Download forecasts as CSV files

## Data Sources

- **Zillow**: Home Value Index (ZHVI) by ZIP code
- **FRED (Federal Reserve)**: Mortgage rates (PMMS)
- **Redfin**: Housing inventory data
- **FHFA**: House Price Index (HPI)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd housing-foresight
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run dashboard/app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. Enter a 5-digit ZIP code (e.g., 08901)
2. Select a model (SARIMAX or XGBoost)
3. Click "Generate Forecast"
4. View the 5-year forecast chart
5. Download results as CSV

## Project Structure

```
housing-foresight/
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── src/
│   ├── ingest/             # Data ingestion modules
│   ├── features/            # Feature engineering
│   ├── models/              # ML models (SARIMAX, XGBoost)
│   └── utils/               # Utility functions
├── data/                    # Cached data (auto-generated)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Technologies

- **Python 3.8+**
- **Streamlit**: Web framework
- **Pandas**: Data manipulation
- **Statsmodels**: SARIMAX time series model
- **XGBoost**: Gradient boosting model
- **Plotly**: Interactive visualizations

## Deployment

This app is ready for deployment on Streamlit Community Cloud:

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Set main file path to: `dashboard/app.py`
6. Deploy!

## License

MIT License

## Author

Mohan Panwar

