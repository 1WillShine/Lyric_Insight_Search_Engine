# main.py
import argparse
import pandas as pd
from src.utils import load_lyrics_csv, ensure_text_column
from src.recommender import calculate_similarity_for_all
from src.tfidf_search import build_tfidf_matrix, top_k_terms_for_doc, search_lyrics
from src.visualize import plot_similarity_hist

def cli_recommend(args):
    df = load_lyrics_csv(args.data)
    ensure_text_column(df, 'lyrics')
    features = args.features.split(",")
    sims = calculate_similarity_for_all(args.uri, df, features)
    df['similarity'] = sims
    top = df.sort_values('similarity', ascending=False).head(args.top)
    print(top[['song','artist','similarity']].to_string(index=False))

def cli_tfidf(args):
    df = load_lyrics_csv(args.data)
    ensure_text_column(df, 'lyrics')
    vect, mat = build_tfidf_matrix(df['lyrics'].tolist(), min_df=1)
    if args.top_terms is not None:
        # print top terms for a given song index
        idx = args.top_terms
        vec = mat[idx]
        top = top_k_terms_for_doc(vec, vect, k=20)
        print(f"Top terms computed for index {idx}:")
        for term, score in top:
            print(f"{term}\t{score:.4f}")
    if args.query is not None:
        res = search_lyrics(args.query, vect, mat, top_n=args.top)
        print("Query results (index, score):")
        for idx, score in res:
            print(idx, score)

def main():
    parser = argparse.ArgumentParser(prog="lyric-insights")
    sub = parser.add_subparsers(dest="cmd")
    # recommend
    p_rec = sub.add_parser("recommend", help="Find similar songs given a song uri/index and numeric features")
    p_rec.add_argument("--data", required=True)
    p_rec.add_argument("--uri", required=True)
    p_rec.add_argument("--features", required=True, help="comma-separated feature names, e.g. Danceability,Energy")
    p_rec.add_argument("--top", type=int, default=10)
    # tfidf
    p_tf = sub.add_parser("tfidf", help="TF-IDF helpers & search")
    p_tf.add_argument("--data", required=True)
    p_tf.add_argument("--top-terms", type=int, default=None, help="show top terms for a given index")
    p_tf.add_argument("--query", default=None)
    p_tf.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    if args.cmd == "recommend":
        cli_recommend(args)
    elif args.cmd == "tfidf":
        cli_tfidf(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
