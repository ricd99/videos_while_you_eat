from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(nn, df):
    distances, _ = nn.kneighbors(df)
    mean_dist = distances.mean()
    median_dist = np.median(distances)

    return mean_dist, median_dist
    