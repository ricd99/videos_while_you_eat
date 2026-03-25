import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer

def train_model(df: pd.DataFrame):
    """
    Trains a Nearest Neighbours model and logs with MLflow.
    """

    # ── Config ─────────────────────────────────────                       
    MAX_FEATURES = 200   
    MIN_DF = 2    
    MAX_DF = 0.95            
    NGRAM_RANGE = (1, 2)    

    N_NEIGHBOURS = 5
    NN_DISTANCE_METRIC = "cosine"               


    df_train, df_test = train_test_split(df, train_size=0.98, random_state=67)
    df_train = df_train.reset_index(drop=True) 
    df_test = df_test.reset_index(drop=True)

    ct = make_column_transformer(
        (CountVectorizer(stop_words="english", max_features=MAX_FEATURES, min_df=MIN_DF, max_df = MAX_DF, ngram_range=NGRAM_RANGE), "text"), #TODO: figure out best hyperparameters
        ("drop", ["channel_id", "channel_name"]),
    )

    df_train_pp = ct.fit_transform(df_train)
    df_test_pp = ct.transform(df_test)
    nn = NearestNeighbors(n_neighbors=N_NEIGHBOURS, metric = NN_DISTANCE_METRIC)        
    

    with mlflow.start_run():
        nn.fit(df_train_pp)

        distances, _ = nn.kneighbors(df_test_pp)
        nn_distances = distances[:, 1:]
        mean_dist = nn_distances.mean()
        median_dist = np.median(nn_distances)

        # ── Log params (settings you chose) ──
        mlflow.log_param("max_features",  MAX_FEATURES)
        mlflow.log_param("min_df",  MIN_DF)
        mlflow.log_param("max_df",  MAX_DF)
        mlflow.log_param("ngram_range",   str(NGRAM_RANGE))
        mlflow.log_param("n_neighbours",   N_NEIGHBOURS)
        mlflow.log_param("metric",        NN_DISTANCE_METRIC)
        mlflow.log_param("train_size",    len(df_train))
        mlflow.log_param("test_size",     len(df_test))

        # ── Log metrics (summary numbers) ──
        mlflow.log_metric("mean_nn_distance",   mean_dist)
        mlflow.log_metric("median_nn_distance", median_dist)


        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print("mlflow run compelte")


