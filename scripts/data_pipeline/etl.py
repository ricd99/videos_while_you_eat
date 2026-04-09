import json
import boto3
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

load_dotenv()

from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features

s3 = boto3.client("s3", region_name="us-west-2")
BUCKET = "ytrec-data-lake"


