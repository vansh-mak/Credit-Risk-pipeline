import joblib
import os

from app.transformers import CreditGarbageClearner

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "credit_risk_pipeline.pkl"
)

model = joblib.load(MODEL_PATH)

def predict(input_df):
    """
    input_df: pandas DataFrame with same columns as training data
    """
    return model.predict_proba(input_df)[:, 1]
