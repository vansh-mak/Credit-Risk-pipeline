# Credit Risk Prediction (ML + FastAPI)

##  Problem
Predict whether a customer will default on credit payments based on financial and demographic features.

##  Approach
- Built an end-to-end ML pipeline using scikit-learn
- Implemented custom data cleaning transformer for handling invalid categorical values
- Used ColumnTransformer for feature-wise preprocessing:
  - Numerical: Imputation + Scaling
  - Nominal: One-Hot Encoding
  - Ordinal: Ordinal Encoding
- Trained Logistic Regression model within pipeline

##  Results
- Accuracy: 0.81
- Used probability-based predictions for decision making

##  Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Pipeline, ColumnTransformer)
- FastAPI
- Joblib
