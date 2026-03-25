import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(df: pd.DataFrame):
    """
    Trains a Nearest Neighbours model and logs with MLflow.
    """