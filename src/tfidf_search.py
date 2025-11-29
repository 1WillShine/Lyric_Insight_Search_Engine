# src/tfidf_search.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple

def build_tfidf_matrix(documents: List[str], min_df: int=1, ngram_range=(1,1)) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Fit a TfidfVectorizer on list of documents and return (vectorizer, matrix)
    matrix shape: (n_docs, n_terms)
    """
    vect = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range, stop_words='english')
    mat = vect.fit_transform(documents)
    return vect, mat

def top_k_terms_for_doc(tfidf_vector, vectorizer: TfidfVectorizer, k: int=10) -> List[Tuple[str, float]]:
    """
    Given a single-row tfidf sparse vector, return top k (term, score) pairs sorted by score desc.
    """
    indices = tfidf_vector.toarray().ravel().argsort()[::-1][:k]
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = tfidf_vector.toarray().ravel()[indices]
    return [(term, float(score)) for term, score in zip(feature_names[indices], scores) if score>0]

def search_lyrics(query: str, vectorizer: TfidfVectorizer, tfidf_matrix, top_n:int=10) -> List[Tuple[int, float]]:
    """
    Rank documents by cosine similarity to the query (using TF-IDF vectorization of the query).
    Returns list of (doc_index, score) sorted desc.
    """
    q_vec = vectorizer.transform([query])
    # cosine similarity between q_vec and every document row
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(q_vec, tfidf_matrix).ravel()
    top_idx = np.argsort(sims)[::-1][:top_n]
    return [(int(i), float(sims[i])) for i in top_idx if sims[i] > 0]
