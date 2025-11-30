# ============================================================
# PAGE 1 — LYRIC SEARCH
# ============================================================
if page == "Lyric Search":
    st.header("🔍 Search Lyrics")

    query = st.text_input("Enter search query")

    if st.button("Search"):
        vect, mat = build_tfidf_matrix(df["Lyrics"].tolist(), min_df=1)
        results = search_lyrics(query, vect, mat, top_n=15)

        st.subheader("Results:")
        for idx, score in results:
            row = df.iloc[idx]

            # FIXED — remove artist, use correct column names
            st.markdown(f"**{row['Song']} — {row['Album']}**")

            st.write(f"Match Score: `{score:.4f}`")

            # FIXED — Lyrics capitalized
            st.write(row["Lyrics"][:250] + "…")
            st.markdown("---")

# ============================================================
# PAGE 2 — SIMILARITY RECOMMENDER
# ============================================================
elif page == "Song Similarity":
    st.header("🎧 Song Similarity Recommender")

    st.write("Enter a song *URI* (or index from the dataset) and choose which numeric features to compute similarity from:")

    uri = st.text_input("Song URI or index (must match 'uri' column in CSV)")

    numeric_cols = [c for c in df.columns if df[c].dtype != "object"]
    features = st.multiselect("Pick numeric audio features:", numeric_cols)

    top_k = st.slider("How many similar songs to show?", 5, 25, 10)

    if st.button("Find Similar Songs"):
        if not features:
            st.error("You must select at least one feature.")
        else:
            sims = calculate_similarity_for_all(uri, df, features)
            df["similarity"] = sims
            top = df.sort_values("similarity", ascending=False).head(top_k)

            st.subheader("Top Recommendations:")

            # FIXED — remove artist
            st.dataframe(top[["Song", "Album", "similarity"]])

# ============================================================
# PAGE 3 — TF-IDF TERM EXPLORER
# ============================================================
else:
    st.header("📚 TF-IDF Top Terms Explorer")

    vect, mat = build_tfidf_matrix(df["Lyrics"].tolist(), min_df=1)

    row_index = st.number_input(
        "Pick a song index:", 
        min_value=0, 
        max_value=len(df) - 1, 
        step=1
    )

    if st.button("Show Top Terms"):
        vec = mat[row_index]
        top = top_k_terms_for_doc(vec, vect, k=20)

        st.subheader("Top TF-IDF Terms for:")

        # FIXED — remove artist
        st.write(f"**{df.iloc[row_index]['Song']} — {df.iloc[row_index]['Album']}**")

        for term, score in top:
            st.write(f"**{term}** — {score:.4f}")
