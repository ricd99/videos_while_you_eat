import joblib
import pandas as pd
import sys, os

MODEL_DIR = "/app/model" if os.path.exists("/app/model") else os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
DATA_PATH = "/app/data/channels_pp.csv" if os.path.exists("/app/data/channels_pp.csv") else os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "channels_pp.csv")
if os.path.exists("/app/src"):
    sys.path.append("/app/src")
else:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocess_data import preprocess_data
from features.build_features import build_features

nn = joblib.load(os.path.join(MODEL_DIR, "nn_model.pkl"))
ct = joblib.load(os.path.join(MODEL_DIR, "column_transformer.pkl"))
df_all = pd.read_csv(DATA_PATH)

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