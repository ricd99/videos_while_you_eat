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
    df = df.drop(["title", "uploads", "country"], axis=1)  # later try keeping country? 
    df = df.reset_index(drop=True)
    CHANNELS_TO_DROP = ["Philosophy Corner", "Psyphos"]
    df = df[~df["channel_name"].isin(CHANNELS_TO_DROP)]

    # topics from wikipedia
    df["topics"] = df["topics"].apply(
        lambda cell: ", ".join(
            url.split("/")[-1].replace("_", " ") for url in cell
        ) if cell is not None else None
    )

    # imputing (manually Doug Woolever because many missing data)
    df.loc[df["channel_name"] == "Doug Woolever", "description"] = "One day when I was watching Spongebob, I noticed something bizarre. This thing was so bizarre that I decided to spend months analyzing every single episode of Spongebob Squarepants in search of answers. What I ended up finding was a completely original and impossibly well-supported conspiracy theory that will destroy everything we have ever been told about Bikini Bottom. The world needs to know the truth. This is the video that will change everything. Watch at your own risk. This video was created for commentary/analysis and to generate discussion through the presentation of research, protected under fair use."
    df.loc[df["channel_name"] == "Doug Woolever", "country"] = "US"
    df.loc[df["channel_name"] == "Doug Woolever", "topics"] = "Entertainment"
    df.loc[df["channel_name"] == "Doug Woolever", "keywords"]  = "TV analysis theory spongebob"

    df["keywords"] = df["keywords"].str.replace("\"", "", regex=False)
    df["topics"] = df["topics"].fillna(df["keywords"])
    df["keywords"] = df["keywords"].fillna(df["topics"])

    # removing description
    df["videos"] = df["videos"].apply(lambda row: " ".join([v["title"] for v in row ]))

    return df
