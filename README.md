# 🚀 Lead–Lag Trading System (ML + XGBoost + LightGBM + GPU)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![ML](https://img.shields.io/badge/Models-XGBoost%20%7C%20LightGBM-orange)
![API](https://img.shields.io/badge/API-FastAPI-green)
![Tests](https://img.shields.io/badge/Tests-Pytest-blue)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 📌 Overview

This project builds a **lead–lag trading strategy** using machine learning models.

It supports:

- Regression → predict next return  
- Classification → predict direction (up/down)  
- Multiple models:
  - Ridge / Logistic (baseline)
  - XGBoost
  - LightGBM  
- Optional GPU acceleration  
- Full backtesting pipeline  
- FastAPI inference API  
- Comprehensive pytest coverage  

---

## 🧠 Problem Statement

Identify whether one asset (X) leads another (Y) and exploit that relationship:

- Predict next move of Y using lagged signals from X and Y  
- Convert predictions into trading positions  
- Evaluate strategy via backtesting  

---

## 🏗 Architecture

```text
Raw Prices
   ↓
Feature Engineering (lags)
   ↓
Model Training (ML / Boosting)
   ↓
Prediction (return or direction)
   ↓
Signal Generation
   ↓
Backtest (PnL + costs)
   ↓
API Deployment
```

---

## ⚙️ Tech Stack

| Layer              | Tools |
|-------------------|------|
| Data Processing    | Pandas, NumPy |
| Modeling           | Scikit-learn, XGBoost, LightGBM |
| API                | FastAPI |
| Testing            | Pytest |
| Visualization      | Matplotlib |

---

## 📂 Project Structure

```text
lead-lag-trading/
├── data/
│   └── sample.csv
├── src/
│   ├── features.py
│   ├── model.py
│   ├── backtest.py
├── artifacts/
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   ├── test_backtest.py
│   ├── test_api.py
│   └── conftest.py
├── api.py
├── train.py
├── main.py
├── config.py
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 🧠 Models Supported

### Regression
- Ridge
- XGBoost Regressor
- LightGBM Regressor

### Classification
- Logistic Regression
- XGBoost Classifier
- LightGBM Classifier

---

## ⚡ GPU Support

Optional GPU acceleration:

```python
"use_gpu": True
```

### Behavior
- Uses GPU if available
- Falls back to CPU automatically
- Works on all machines (no failure)

---

## 🧪 Testing (Pytest)

Run:

```bash
pytest -v
```

### Coverage

- Feature generation
- Model training (CPU + GPU fallback)
- Backtesting logic
- API endpoints

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Train model

```bash
python train.py
```

---

### 3. Run backtest

```bash
python main.py
```

---

### 4. Start API

```bash
python -m uvicorn api:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 📈 Backtesting

- Converts predictions → positions
- Applies transaction costs
- Outputs:
  - Equity curve
  - Positions
  - Metrics

---

## 🔌 API Usage (Swagger Examples)

Go to:

```
http://127.0.0.1:8000/docs
```

### Regression Example

```json
{
  "features": {
    "X": 103.4,
    "Y": 201.2,
    "X_ret": 0.18,
    "Y_ret": -0.06,
    "X_lag_1": 0.12,
    "X_lag_2": 0.08,
    "X_lag_3": -0.04,
    "X_lag_4": 0.03,
    "X_lag_5": 0.01,
    "Y_lag_1": -0.02,
    "Y_lag_2": 0.01,
    "Y_lag_3": 0.03,
    "Y_lag_4": -0.01,
    "Y_lag_5": 0.02
  }
}
```

---

### Classification Example

```json
{
  "features": {
    "X": 104.1,
    "Y": 200.8,
    "X_ret": 0.22,
    "Y_ret": 0.04,
    "X_lag_1": 0.15,
    "X_lag_2": 0.09,
    "X_lag_3": 0.02,
    "X_lag_4": -0.01,
    "X_lag_5": 0.03,
    "Y_lag_1": 0.01,
    "Y_lag_2": -0.02,
    "Y_lag_3": 0.04,
    "Y_lag_4": 0.00,
    "Y_lag_5": 0.02
  }
}
```

---

### Expected Responses

#### Regression

```json
{
  "task_type": "regression",
  "model_type": "xgboost_reg",
  "prediction": 0.0142
}
```

#### Classification

```json
{
  "task_type": "classification",
  "model_type": "lightgbm_clf",
  "prediction": 1,
  "probability_up": 0.73
}
```

---

## 🔥 Key Highlights

- Lead–lag trading strategy  
- Regression + classification modeling  
- XGBoost & LightGBM integration  
- GPU-aware training with fallback  
- Full backtesting engine  
- API deployment  
- Strong test coverage  

---

## 🧠 Talking Points

- Built ML-driven trading system using lead–lag signals  
- Compared linear vs boosting models  
- Implemented regression vs classification strategies  
- Added GPU-aware training with fallback logic  
- Designed backtesting framework with transaction costs  
- Created full test suite for robustness  

---

## 📌 Author

Machine Learning + Quant Trading Portfolio Project
