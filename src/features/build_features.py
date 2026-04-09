import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering pipeline on training data
    """
    df = df.copy()
    fields = [
        df["description"],
        df["topics"],
        df["keywords"],
        df["videos"]
    ]

    df["text"] = [str(f) for f in fields if f pd.notna(f)]
    df = df.drop(["description", "topics", "keywords", "videos"], axis=1)

    return df;