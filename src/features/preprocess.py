import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    df = df.copy()
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Missing")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df
