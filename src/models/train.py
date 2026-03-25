import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer

def train_model(df: pd.DataFrame, n_neighbours: int = 5):
    """
    Trains a Nearest Neighbours model and logs with MLflow.
    """

    df_train, df_test = train_test_split(df, train_size=0.98, random_state=67)
    df_train = df_train.reset_index(drop=True) 
    df_test = df_test.reset_index(drop=True)

    ct = make_column_transformer(
    (CountVectorizer(stop_words="english", max_features=200, min_df=2, max_df = 0.95), "text"), #TODO: figure out best hyperparameters
    ("drop", ["channel_id", "channel_name"]),
    )

    df_train_pp = ct.fit_transform(df_train)
    df_big_test_pp = ct.transform(df_test)
    nn = NearestNeighbors(n_neighbors=n_neighbours, metric="cosine")
    

    with mlflow.start_run():
        nn.fit(df_train_pp)

        distances, _ = nn.kneighbors(df_test)
        nn_distances = distances[:, 1:]
        mean_dist = nn_distances.mean()
        median_dist = np.median(nn_distances)


