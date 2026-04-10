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

    df["text"] = df.apply(
        lambda row: "\n".join(
            str(x) for x in [                # str(x) to be safe
                row["description"],
                row["topics"],
                row["keywords"],
                row["videos"]
            ] if pd.notna(x)
        ),
        axis=1 #TODO sequential operation to check for Nans
    )
    
    df = df[df["text"].str.strip() != ""]
    df = df.drop(["description", "topics", "keywords", "videos"], axis=1)

    return df;