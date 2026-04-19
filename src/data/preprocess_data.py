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
    df = df.drop(["country", "subscriber_count", "video_count", "months_since_publish", "flags"], axis=1, errors="ignore")  # later try keeping country?, errors=ignore does nto raise error of column no exist 
    df = df.reset_index(drop=True)

    # topics - handle both wikipedia URLs and plain topic names from YouTube API
    def _extract_topic(cell):
        if cell is None:
            return None
        if isinstance(cell, list):
            return ", ".join(
                url.split("/")[-1].replace("_", " ") if "/" in url else url
                for url in cell
            )
        return str(cell)

    df["topics"] = df["topics"].apply(_extract_topic)

    df["keywords"] = df["keywords"].str.replace("\"", "", regex=False)  
    df["topics"] = df["topics"].fillna(df["keywords"])                      #TODO: remove this cross imputing (as it is all combined into text later anyway?)
    df["keywords"] = df["keywords"].fillna(df["topics"])

    # removing description
    df["videos"] = df["videos"].apply(lambda row: " ".join([v["title"] for v in row]) if isinstance(row, list) else None) # crashes without isinstance check as v in row would iterate over None

    return df
