import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for youtube dataset
    
    - dropping unused columns and AI slop channels
    - extracting topics from wikipedia link
    - imputing missing values
    - removing video description from "video" feature
    """
    # dropping
    df = df.drop(["country"], axis=1, errors="ignore")  # later try keeping country?, errors=ignore does nto raise error of column no exist 
    df = df.reset_index(drop=True)

    # topics from wikipedia
    df["topics"] = df["topics"].apply(
        lambda cell: ", ".join(
            url.split("/")[-1].replace("_", " ") for url in cell
        ) if cell is not None else None
    )

    df["keywords"] = df["keywords"].str.replace("\"", "", regex=False)  #TODO: remove this cross imputing (as it is all combined into text later anyway?)
    df["topics"] = df["topics"].fillna(df["keywords"])
    df["keywords"] = df["keywords"].fillna(df["topics"])

    # removing description
    df["videos"] = df["videos"].apply(lambda row: " ".join([v["title"] for v in row ]))

    return df
