import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads JSON data into a pandas DataFrame.
    
    Args:
        file_path(str): Path to the JSON file

    Returns: pd.DataFrame: Loaded dataset.
    """

    if not (os.path.exists(file_path)):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_json(file_path)