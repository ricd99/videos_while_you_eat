"""
Training module for Nearest Neighbors model using sentence embeddings.
This replaces CountVectorizer with SentenceTransformer for better semantic
understanding of channel text (descriptions, topics, keywords, video titles).
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from src.embedding import batch_encode

def train_model(df_train: pd.DataFrame, params: dict) -> tuple[NearestNeighbors, np.ndarray, pd.DataFrame]:                       
    """
    Train NearestNeighbors model on sentence embeddings.
    Args:
        df_train: DataFrame with 'text' column containing channel descriptions
        params: Dictionary with 'n_neighbors' and 'metric' keys
    Returns:
        tuple: (fitted_nn_model, embedding_matrix, df_with_embeddings)
    
    Example:
        >>> df = load_data("data/processed/ve_channels/ve_with_features.json")
        >>> df = preprocess_data(df)
        >>> df = build_features(df)
        >>> nn, embeddings, df_emb = train_model(df, {"n_neighbors": 15, "metric": "cosine"})
    """

    if "text" not in df_train.columns:
        raise ValueError("df_train must have 'text' column from build_features()")
    
    texts = df_train["text"].fillna("").tolist()

    embeddings = batch_encode(texts)

    nn = NearestNeighbors(n_neighbors=params["n_neighbors"], metric=params["metric"])
    nn.fit(embeddings)

    df_lookup = df_train[["channel_name", "channel_id"]].reset_index(drop=True)
    
    return nn, embeddings, df_lookup