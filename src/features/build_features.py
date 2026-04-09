import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering pipeline on training data
    """
    df = df.copy()
    df["text"] = df["description"].astype(str) + "\n" + df["topics"].astype(str) + "\n" + df["keywords"].astype(str) + "\n" + df["videos"].astype(str) 
    df = df.drop(["description", "topics", "keywords", "videos"], axis=1)

    return df;