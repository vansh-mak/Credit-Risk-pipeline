from fastapi import FastAPI
from app.model import predict
import pandas as pd
from app.schema import CreditInput

app = FastAPI()

@app.post("/predict")
def predict_credit(data: CreditInput):
    df = pd.DataFrame([data.dict()])
    prob = predict(df)
    return {"default_probability": float(prob[0])}
