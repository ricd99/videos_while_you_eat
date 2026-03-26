import os, sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features

RAW = "data/processed/ve_channels/ve_with_features.json"
OUT = "data/preprocessed/channels_pp.csv"

df = pd.read_json(RAW)

df = preprocess_data(df)

# data sanity checks here?

df_pp = build_features(df)


os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_pp.to_csv(OUT, index=False)
print(f"Processed data saved to {OUT} | Shape: {df_pp.shape}")