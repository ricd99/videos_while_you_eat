import great_expectations as ge
from typing import Tuple, List

def validate_youtube_data(df) -> Tuple(bool, List[str]):
    """
    Comprehensive data validation for youtube dataset using Great Expectations

    This function implements data quality checls that must pass before model training. 
    """