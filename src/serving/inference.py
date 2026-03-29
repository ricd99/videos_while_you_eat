import joblib
import pandas as pd

# load artifacts saved by run_pipeline.py
nn = joblib.load("/app/model/nn_model.pkl")
ct = joblib.load("/app/model/column_transformer.pkl")

# load the original dataset so we can return channel names
df_all = pd.read_csv("/app/data/channels_pp.csv")

def predict(input_dict: dict):
    df = pd.DataFrame([input_dict])
    df_transformed = ct.transform(df)
    distances, indices = nn.kneighbors(df_transformed)
    
    results = df_all.iloc[indices[0]][["channel_name", "channel_id"]].copy()
    results["similarity_score"] = distances[0]
    return results.to_dict(orient="records")