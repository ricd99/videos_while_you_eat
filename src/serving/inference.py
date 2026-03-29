import joblib
import pandas as pd
import sys, os
sys.path.append("/app/src")

from data.preprocess_data import preprocess_data
from features.build_features import build_features

nn = joblib.load("/app/model/nn_model.pkl")
ct = joblib.load("/app/model/column_transformer.pkl")
df_all = pd.read_csv("/app/data/channels_pp.csv")

def predict(input_dict: dict):
    df = pd.DataFrame([input_dict])
    
    # same steps as training pipeline
    df = preprocess_data(df)
    df = build_features(df)
    df_transformed = ct.transform(df)
    
    distances, indices = nn.kneighbors(df_transformed)
    
    results = df_all.iloc[indices[0]][["channel_name", "channel_id"]].copy()
    results["similarity_score"] = distances[0]
    return results.to_dict(orient="records")