# src/recommender.py
import numpy as np
import pandas as pd
from typing import List

def get_feature_values(uri_or_idx, df: pd.DataFrame, feature_list: List[str]):
    """
    Accept either an index or a unique identifier (uri) string to find the row in df.
    Returns a numpy array of the selected features (floats).
    """
    if isinstance(uri_or_idx, str):
        # assume 'uri' column
        row = df.loc[df['uri'] == uri_or_idx]
        if row.empty:
            raise KeyError(f"URI '{uri_or_idx}' not found")
        row = row.iloc[0]
    else:
        row = df.loc[uri_or_idx]
    vals = row[feature_list].astype(float).to_numpy()
    return vals

def calculate_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity with safe fallback for zero vectors.
    """
    if vec_a.shape != vec_b.shape:
        raise ValueError("Vectors must be same shape")
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)

def calculate_similarity_for_all(input_uri, song_df: pd.DataFrame, feature_list: List[str]) -> np.ndarray:
    """
    Compute similarity between the chosen song (input_uri) and every song in song_df using feature_list.
    Returns an array of similarity scores aligned with song_df.index
    """
    target_vec = get_feature_values(input_uri, song_df, feature_list)
    similarity_scores = np.empty(len(song_df), dtype=float)
    for i, (_, row) in enumerate(song_df.iterrows()):
        vec = row[feature_list].astype(float).to_numpy()
        similarity_scores[i] = calculate_similarity(target_vec, vec)
    return similarity_scores
