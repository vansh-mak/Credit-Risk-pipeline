from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CreditGarbageClearner(BaseEstimator, TransformerMixin):
    def __init__(self, pay_cols=None):
        self.pay_cols = pay_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['EDUCATION'] = X['EDUCATION'].replace([0, 5, 6], 4)
        X['MARRIAGE'] = X['MARRIAGE'].replace(0, 3)

        pay_cols = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
        for col in pay_cols:
            X[col] = X[col].clip(-1, 4)

        return X