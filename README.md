# Apple Stock Price Prediction (1980–2022) Using GRU Deep Learning

This project implements, evaluates, and deploys a deep learning–based stock price prediction system using a Stacked GRU architecture trained on Apple Inc. (AAPL) historical stock data spanning over 40 years.

The system includes a complete time series preprocessing pipeline and a production-style inference interface with multi-day autoregressive forecasting.

**Live Demo:** [[https://huggingface.co/spaces/your-username/your-space-name](#)](https://huggingface.co/spaces/shihabkarol/Apple-Stock-Price-Prediction)  

---

## Problem Statement

Build a regression model to predict Apple's future **Close Price (USD)** using historical OHLCV data, and deploy it as a real-time multi-day forecasting system.

---

## Model Architecture

- MinMaxScaler Preprocessing (5 features)
- GRU Layer 1 (50 units, return_sequences=True)
- Dropout (0.2)
- GRU Layer 2 (50 units)
- Dropout (0.2)
- Dense Output Layer (Linear activation)
- Loss Function: Mean Squared Error
- Optimizer: Adam
- Look-back Window: 60 days
- Total Parameters: 23,901

The stacked GRU architecture was selected to capture long-range sequential dependencies in financial time series data.

---

## Model Performance

| Metric | Score |
|--------|--------|
| MAE | **$3.34** |
| RMSE | **$6.12** |
| MAPE | **3.447%** |
| Prediction Accuracy | **96.55%** |

The model tracks Apple's price trajectory closely through 2014–2019. The widening gap post-2020 reflects the COVID-driven bull run — a market anomaly beyond historical training patterns, not a model deficiency.

---

## Time Series Preprocessing Pipeline

The following preprocessing steps were implemented:

- OHLCV feature selection (Open, High, Low, Close, Volume)
- Min-Max scaling to [0, 1] range
- 60-day sliding window sequence creation
- 80/20 chronological train/test split
- Inverse transformation for interpretable predictions
- Full dataset retraining for production deployment

The scaler and trained model were serialized to ensure reproducible deployment.

---

## Deployment & Inference System

The model is deployed using Hugging Face Spaces with:

- CSV upload interface for historical stock data
- Autoregressive multi-day forecasting (up to 30 days)
- Day-by-day price prediction table with % change
- Actual vs Forecast line chart
- Forecast summary with overall direction and range
- Stateless prediction pipeline

Model saved in `.h5` format  
Scaler serialized via `joblib`

---

## Tableau Performance Dashboard

A 5-chart interactive dashboard was built to evaluate model behavior:

- Actual vs Predicted Close Price (Line Chart)
- Residuals Over Time (Bar Chart)
- Scatter Plot — Actual vs Predicted
- Model KPIs — MAE, RMSE, MAPE
- Error Distribution (Histogram)

---

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- Gradio
- Hugging Face Spaces
- Tableau Public

---

## Project Highlights

- End-to-end ML pipeline from raw data to live deployment
- Reproducible model artifact serialization
- Autoregressive multi-step forecasting capability
- Interactive Tableau dashboard for model performance analysis
- Evaluation-aware design with separate test set and full dataset retraining

---

## Potential Improvements

- Add technical indicators (RSI, MACD, Bollinger Bands) as features
- Implement multi-stock generalization beyond AAPL
- Add baseline comparison (ARIMA, Prophet)
- Integrate attention mechanism for interpretability
- Expose REST API using FastAPI

---

## Disclaimer

> This project is for **educational and portfolio purposes only**. Predictions are **not financial advice** and should not be used for investment decisions.
