import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer

def train_model(df_train, df_test, params):
    """
    Trains a Nearest Neighbours model and logs with MLflow.
    Returns fitted nn and column transformer
    """

    ct = make_column_transformer(
        (CountVectorizer(stop_words="english", 
                         max_features=params["max_features"], 
                         min_df=params["min_df"], 
                         max_df = params["max_df"], 
                         ngram_range=(1, params["ngram_max"])), 
                         "text"), 
        ("drop", ["channel_id", "channel_name"]),
    )

    df_train_pp = ct.fit_transform(df_train)
    nn = NearestNeighbors(n_neighbors=5, metric = params["metric"])    
    nn.fit(df_train_pp)    

    return nn, ct


