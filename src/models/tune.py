import optuna 
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer

def tune_model(df_train, df_test):
    """
    Tunes CountVectorizer + NearestNeighbors using Optuna
    """

    def objective(trial):

        max_features = trial.suggest_int("max_features", 100, 2000, step=100)
        min_df       = trial.suggest_int("min_df", 1, 5)
        max_df       = trial.suggest_float("max_df", 0.7, 1.0, step=0.05)
        ngram_max    = trial.suggest_int("ngram_max", 1, 3) 
        metric       = trial.suggest_categorical("metric", ["cosine", "euclidean"])

        ct = make_column_transformer(
            (CountVectorizer(
                stop_words="english",
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                ngram_range=(1, ngram_max)
            ), "text"),
            ("drop", ["channel_id", "channel_name"]),
        )

        try:
            df_train_pp = ct.fit_transform(df_train)
            df_test_pp  = ct.transform(df_test)
        except ValueError:
            # Some param combos produce an empty vocabulary
            return float("inf")

        nn = NearestNeighbors(n_neighbors=5, metric=metric)
        nn.fit(df_train_pp)

        distances, _ = nn.kneighbors(df_test_pp)
        mean_dist = distances.mean()

        return mean_dist 

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    return study.best_params