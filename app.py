# app.py - Lyric Insight Search Engine (Streamlit)
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

# sklearn imports (TF-IDF + similarity)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------- Helpers ----------
DATA_PATH = Path("data/lyrics.csv")

st.set_page_config(page_title="Lyric Insight Search Engine", layout="wide")

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Dataset not found at `{path}`. Put your CSV in a folder called `data/` and name it `lyrics.csv`.")
        st.stop()
    df = pd.read_csv(path)
    # normalize whitespace in column names
    df.columns = [c.strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, target: str) -> Optional[str]:
    """Return actual column name in df matching target case-insensitive, else None."""
    target_lower = target.lower()
    for c in df.columns:
        if c.lower() == target_lower:
            return c
    return None

@st.cache_data(show_spinner=False)
def build_tfidf_matrix(corpus: List[str], min_df: int = 1):
    vect = TfidfVectorizer(min_df=min_df, stop_words="english")
    mat = vect.fit_transform(corpus)
    return vect, mat

def top_k_terms_for_doc(row_vector, vect: TfidfVectorizer, k: int = 20) -> List[Tuple[str, float]]:
    # row_vector is a sparse vector
    if hasattr(row_vector, "toarray"):
        arr = row_vector.toarray().ravel()
    else:
        arr = np.asarray(row_vector).ravel()
    top_idx = arr.argsort()[::-1][:k]
    terms = vect.get_feature_names_out()
    return [(terms[i], float(arr[i])) for i in top_idx if arr[i] > 0]

def search_lyrics(query: str, vect: TfidfVectorizer, mat, top_n: int = 10) -> List[Tuple[int, float]]:
    q_vec = vect.transform([query])
    sims = cosine_similarity(q_vec, mat).ravel()
    top_idx = sims.argsort()[::-1][:top_n]
    return [(int(idx), float(sims[idx])) for idx in top_idx if sims[idx] > 0]

def safe_get_numeric_cols(df: pd.DataFrame) -> List[str]:
    # numeric but not boolean, not index-like
    numerics = df.select_dtypes(include=[np.number]).columns.tolist()
    return numerics

def compute_feature_similarity_matrix(df: pd.DataFrame, feature_cols: List[str]):
    arr = df[feature_cols].fillna(0).astype(float).to_numpy()
    # normalize
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    arrn = arr / denom
    sim = arrn.dot(arrn.T)
    return sim

# --------- Load data & validate columns ----------
df = load_dataset(DATA_PATH)

col_song = find_col(df, "song")
col_album = find_col(df, "album")
col_lyrics = find_col(df, "lyrics")

if col_song is None or col_lyrics is None:
    st.error("Required columns not found. Dataset must include at least these columns (case-insensitive): `Song`, `Lyrics`. "
             f"Found columns: {list(df.columns)}")
    st.stop()

# For convenience, create lowercase versions for internal use
df = df.reset_index(drop=True)
df["_index"] = df.index  # explicit index column
# show small preview in sidebar
with st.sidebar:
    st.markdown("### Dataset preview")
    st.dataframe(df[[col_song, col_lyrics]].head(5), use_container_width=True)

# --------- Build TF-IDF upfront (cached) ----------
with st.spinner("Building TF-IDF matrix..."):
    try:
        vect, mat = build_tfidf_matrix(df[col_lyrics].astype(str).tolist(), min_df=1)
    except Exception as e:
        st.error("Error building TF-IDF matrix. Make sure scikit-learn is installed and dataset contains textual lyrics.")
        st.stop()

# --------- App navigation ----------
pages = ["Lyric Search", "Song Similarity", "TF-IDF Top Terms", "Song Index"]
page = st.radio("Navigate", pages, index=0, horizontal=True)

# Utility: human label for song option
def song_option(i: int) -> str:
    title = str(df.at[i, col_song])
    album = str(df.at[i, col_album]) if col_album else ""
    display = f"{i} — {title}"
    if album:
        display += f" ({album})"
    return display

# ---------- PAGE: Lyric Search ----------
if page == "Lyric Search":
    st.header("🔍 Lyric Search — press Enter to run")

    query = st.text_input("Enter search query (press Enter to submit)")

    if query:
        results = search_lyrics(query, vect, mat, top_n=20)
        if not results:
            st.info("No results (similarity scores are zero). Try a different query.")
        else:
            st.subheader("Results")
            for idx, score in results:
                if idx < 0 or idx >= len(df):
                    continue
                row = df.iloc[idx]
                song_title = row[col_song]
                album = row[col_album] if col_album else ""
                st.markdown(f"**{song_title}** — Index `{idx}` {f'• {album}' if album else ''}")
                st.write(f"**Score:** `{score:.4f}`")
                snippet = str(row[col_lyrics])[:400].replace("\n", " ")
                st.write(snippet + ("…" if len(str(row[col_lyrics])) > 400 else ""))
                st.markdown("---")


# ---------- PAGE: Song Similarity ----------
elif page == "Song Similarity":
    st.header("🎧 Song Similarity Recommender")
    st.markdown(
        "Select a seed song (by typing/selecting title) and choose numeric audio features to compute similarity. "
        "If your dataset doesn't include audio features (danceability, energy, etc.), add them as columns to `data/lyrics.csv`."
    )

    # build options list "index — title (album)" for selectbox
    options = [song_option(i) for i in range(len(df))]
    selected = st.selectbox("Select seed song (type to search titles)", options, index=0, help="Start typing a title to filter")

    # parse index
    try:
        seed_idx = int(selected.split("—")[0].strip())
    except Exception:
        st.error("Unable to parse selected song index. Please choose from the dropdown.")
        st.stop()

    numeric_cols = safe_get_numeric_cols(df)
    if not numeric_cols:
        st.warning(
            "No numeric feature columns detected in the dataset. Similarity requires numeric audio features (e.g., Danceability, Energy, Valence). "
            "Add columns to `data/lyrics.csv` and re-upload. Detected columns: " + ", ".join(df.columns)
        )
    else:
        chosen = st.multiselect("Numeric features to use for similarity", numeric_cols, default=numeric_cols[:3])
        top_k = st.slider("How many similar songs to show?", min_value=3, max_value=30, value=10)
        if st.button("Find Similar Songs"):
            if not chosen:
                st.error("Select at least one numeric feature.")
            else:
                sim_mat = compute_feature_similarity_matrix(df, chosen)
                sims = sim_mat[seed_idx]
                ranked_idx = np.argsort(sims)[::-1]
                # exclude seed itself at top
                ranked_idx = [i for i in ranked_idx if i != seed_idx][:top_k]
                results_df = df.iloc[ranked_idx][[col_song] + ([col_album] if col_album else [])].copy()
                results_df["similarity"] = sims[ranked_idx]
                st.subheader("Top matches")
                st.dataframe(results_df.reset_index().rename(columns={"_index":"index"}), use_container_width=True)

# ---------- PAGE: TF-IDF TOP TERMS ----------
elif page == "TF-IDF Top Terms":
    st.header("📚 TF-IDF Top Terms Explorer")
    st.markdown("Pick a song (by title) to view its highest TF-IDF terms.")

    options = [song_option(i) for i in range(len(df))]
    sel = st.selectbox("Pick a song", options, index=0)
    try:
        sel_idx = int(sel.split("—")[0].strip())
    except Exception:
        st.error("Unable to parse selection. Please choose from the dropdown.")
        st.stop()

    k = st.slider("How many top terms to show?", min_value=5, max_value=50, value=20)
    if st.button("Show Top Terms"):
        vec = mat[sel_idx]
        top_terms = top_k_terms_for_doc(vec, vect, k=k)
        st.subheader(f"Top TF-IDF terms for `{df.at[sel_idx, col_song]}` (Index {sel_idx})")
        if not top_terms:
            st.info("No high-scoring TF-IDF terms for this document.")
        else:
            for term, score in top_terms:
                st.write(f"**{term}** — {score:.4f}")

# ---------- PAGE: SONG INDEX ----------
elif page == "Song Index":
    st.header("📋 Song Index (copy indices for CLI or analysis)")
    st.markdown("This table shows the mapping between dataset index and song title. Use indices in other pages if you prefer numeric selection.")
    index_table = df[ [ "_index", col_song ] + ([col_album] if col_album else []) ].copy()
    index_table = index_table.rename(columns={"_index": "index", col_song: "song", col_album: "album"} if col_album else {"_index":"index", col_song:"song"})
    st.dataframe(index_table.reset_index(drop=True), use_container_width=True)
