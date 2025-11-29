# src/utils.py
import pandas as pd
import os
from typing import Optional

def load_lyrics_csv(path: str) -> pd.DataFrame:
    """
    Load dataset that contains song lyrics and metadata.
    Expected minimal columns: ['song', 'artist', 'lyrics', 'uri'] or similar.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Put your dataset in the 'data/' folder.")
    df = pd.read_csv(path)
    return df

def ensure_text_column(df: pd.DataFrame, col: str = "lyrics"):
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame. Found columns: {list(df.columns)}")
    df[col] = df[col].fillna("").astype(str)
    return df
