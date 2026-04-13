# Data Scientist Web App

## Overview

A comprehensive data science web application built with Streamlit. Users can upload datasets (CSV/Excel), automatically clean data, perform analysis, generate interactive visualizations, and run basic ML predictions.

## Stack

- **Frontend/Backend**: Streamlit (Python)
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **File Support**: openpyxl (Excel), xlrd

## Project Structure

```
app.py                    # Main Streamlit application
modules/
  data_loader.py          # File upload and dataset info display
  data_cleaner.py         # Auto and manual data cleaning (duplicates, missing values, outliers)
  data_analysis.py        # Summary stats, correlations, distributions, trends
  visualizations.py       # Auto-generated and custom chart builder
  insights.py             # Automated insight generation
  ml_module.py            # Basic ML prediction (classification & regression)
.streamlit/config.toml    # Streamlit configuration (dark theme)
```

## Key Commands

- `streamlit run app.py --server.port 5000` — run the application

## Features

- **Data Preview**: Dataset shape, column types, missing values summary
- **Data Cleaning**: Auto-clean (remove duplicates, fill missing, detect outliers) or manual controls
- **Analysis**: Summary statistics, correlation heatmaps, distributions, trend detection
- **Visualization**: Auto-generated charts + custom builder (scatter, line, bar, pie, box, violin, histogram, area, sunburst)
- **Insights**: Automated detection of correlations, skewness, outliers, imbalances
- **ML Prediction**: Classification (Logistic Regression, Random Forest) and Regression (Linear, Random Forest)
- **Downloads**: Export cleaned data as CSV or Excel
