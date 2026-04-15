from fastapi import FastAPI
import numpy as np

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/predict")
def predict(data: list):
    return {"prediction": float(np.mean(data))}