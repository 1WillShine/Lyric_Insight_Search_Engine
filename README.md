# 🎵 Lyrics with Intelligence 
### A TF-IDF Lyric Search Engine & Audio-Feature Song Recommender  
A standalone, production-ready version of my original Data Science class project(DSC10) — converted from a Jupyter notebook into a clean Python package with a lightweight Streamlit web app.

---

## 📌 Overview
**Lyric Insights** is a modular toolkit for exploring, analyzing, and searching song lyrics.  
It includes:

### ✔️ TF-IDF Lyric Search  
- Query lyrics (e.g., *"heartbreak", "summer nights", "dancing"*).  
- Rank songs by cosine similarity.  
- Extract top meaningful words per song.

### ✔️ Audio-Feature Based Song Recommender  
- Uses numeric features like *Danceability, Energy, Valence, Acousticness*, etc.  
- Computes cosine similarity to recommend similar songs.  
- Powered by clean modular Python code.

### ✔️ Visualization Helpers  
- Distribution plots  
- Keyword summaries

### ✔️ Streamlit Web App  
- Fully runnable on **Streamlit Cloud** (or locally).  
- Search lyrics and view recommendations interactively.


---

## 📦 Installation
### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/lyric-insights.git
cd lyric-insights
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. pip install -r requirements.txt
```bash
pip install -r requirements.txt
```


## 🚀 Running the Streamlit App
## Run locally
```bash
streamlit run app.py
```

### Deploy to Streamlit Cloud

  1. Push to GitHub

  2. Go to https://streamlit.io/cloud

  3. Deploy → pick the repo

  4. Set working directory to repository root

  5. No additional config needed









