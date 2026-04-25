from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("decision_tree_regressor_model.joblib")

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: list):
    arr = np.array(data).reshape(1, -1)
    prediction = model.predict(arr)
    return {"prediction": float(prediction[0])}
