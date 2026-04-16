# 🚀 Lead–Lag Trading System (End-to-End MLE Project)

An end-to-end machine learning system that models **intraday lead–lag relationships** between two time series (X → Y), generates trading signals, and exposes predictions via a production-ready API.

---

## 📌 Problem

In many financial markets, one asset (X) leads another (Y) with a short delay.  
This project builds a system to:

- Detect lead–lag relationships
- Predict short-term returns of Y using X
- Generate trading signals
- Backtest strategy performance
- Serve predictions via API

---

## 🏗 Architecture

Data → Feature Engineering → Model Training → Backtest → API Serving

---

## ⚙️ Tech Stack

- Data Processing: Pandas, NumPy  
- Modeling: Scikit-learn (Ridge Regression)  
- Visualization: Matplotlib  
- API: FastAPI  
- Server: Uvicorn  
- Testing: Pytest  
- Serialization: Joblib  

---

## 📂 Project Structure

lead-lag-trading/
├── data/
│   └── sample.csv
├── src/
│   ├── features.py
│   ├── model.py
│   └── backtest.py
├── artifacts/
│   ├── model.joblib
│   └── feature_columns.joblib
├── tests/
│   └── test_basic.py
├── config.py
├── train.py
├── main.py
├── api.py
├── requirements.txt
└── README.md

---

## 📊 Feature Engineering

- Lagged returns of X and Y
- Autoregressive features
- Target = next-period return of Y

Example:
X_lag_1, X_lag_2, ..., X_lag_k  
Y_lag_1, Y_lag_2, ..., Y_lag_k  

---

## 🧠 Model

- Ridge Regression (L2 regularization)
- Predicts:
  Y_return(t+1)

---

## 📈 Backtesting Logic

- Signal:
  - Long if prediction > threshold
  - Short if prediction < -threshold
- Includes:
  - Transaction costs
  - Position changes
  - Cumulative PnL

---

## ▶️ How to Run

### 1. Install dependencies
pip install -r requirements.txt

---

### 2. Generate data
python generate_data.py

---

### 3. Train model
python train.py

---

### 4. Run backtest
python main.py

---

### 5. Run API
python -m uvicorn api:app --reload

Open:
http://127.0.0.1:8000/docs

---

## 🔌 API Usage

### POST /predict

Input:
{
  "features": {
    "X": 100,
    "Y": 90,
    "X_ret": 0.2,
    "Y_ret": -0.1,
    "X_lag_1": 0.1,
    "X_lag_2": 0.05
  }
}

Output:
{
  "prediction": 0.0123
}

---

## 🧪 Tests

pytest

---

## 📊 Example Output

- Equity curve visualization
- Strategy PnL over time
- Model predictions vs actual returns

---

## 🔥 Key Highlights (MLE Focus)

- End-to-end pipeline (data → model → API)  
- Realistic time-series simulation  
- Feature engineering for lag structures  
- Transaction-cost-aware backtesting  
- Model serialization and serving  
- Production-style API with FastAPI  

---

## 🚀 Future Improvements

- Walk-forward validation (rolling windows)
- Hyperparameter tuning
- XGBoost / deep learning models
- Real-time streaming (Kafka)
- Docker + CI/CD pipeline
- Feature store integration

---

## 🧠 Interview Talking Points

- Designed a lead–lag predictive system
- Built feature pipeline for time-series modeling
- Implemented backtesting with realistic constraints
- Deployed model as low-latency API
- Demonstrated full ML lifecycle (MLOps-ready)

---

## 📌 Author

Built as part of Machine Learning Engineering portfolio projects.
