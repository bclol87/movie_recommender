import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# CONFIG
# ===============================
API_KEY = "3eb39709869b67fd15b086e095c5cbec"

st.set_page_config(page_title="CineMatch", page_icon="🍿", layout="wide")

# ===============================
# PREMIUM CSS
# ===============================
st.markdown("""
<style>

/* Background Gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}

/* Hide Streamlit Menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Titles */
.main-title {
    font-size: 4rem;
    font-weight: 900;
    text-align: center;
    background: -webkit-linear-gradient(#E50914, #ff6a00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title {
    text-align: center;
    font-size: 1.2rem;
    color: #dddddd;
    margin-bottom: 2rem;
}

/* Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.37);
    transition: 0.3s;
}

.glass-card:hover {
    transform: scale(1.03);
}

/* Score Badge */
.score-badge {
    background: linear-gradient(45deg, #00c853, #64dd17);
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 0.9rem;
    display: inline-block;
    margin-bottom: 10px;
}

/* Section Header */
.section-header {
    font-size: 1.8rem;
    font-weight: bold;
    margin-top: 2rem;
    margin-bottom: 1rem;
    border-left: 6px solid #E50914;
    padding-left: 10px;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_and_prep_data():
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=column_names)

    movie_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown',
                  'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                  'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                         header=None, names=movie_cols)

    merged = pd.merge(ratings, movies[['item_id', 'title']], on='item_id')
    movie_matrix = merged.pivot_table(index='user_id',
                                      columns='title',
                                      values='rating')

    ratings_count = merged.groupby('title')['rating'].count().to_frame()
    ratings_count.rename(columns={'rating': 'num_of_ratings'}, inplace=True)

    genre_data = movies.drop(columns=['item_id', 'release_date',
                                      'video_release_date', 'imdb_url', 'unknown'])
    genre_data = genre_data.groupby('title').max().fillna(0).astype(int)

    genre_list = ['All Genres'] + list(genre_data.columns)

    return ratings, movie_matrix, ratings_count, genre_data, sorted(movies['title'].unique()), genre_list


ratings, movie_matrix, ratings_count, genre_data, movie_list, genre_list = load_and_prep_data()

# ===============================
# TMDB HELPERS
# ===============================
def clean_movie_title(title):
    clean_title = re.sub(r'\s*\(\d{4}\)', '', title).strip()
    if clean_title.endswith(', The'):
        clean_title = 'The ' + clean_title[:-5]
    return clean_title


@st.cache_data(ttl=86400)
def fetch_movie_details(movie_title):
    search_title = clean_movie_title(movie_title)
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={search_title}"

    try:
        response = requests.get(url)
        data = response.json()
        if data['results']:
            movie = data['results'][0]
            poster = movie.get('poster_path')
            poster_url = f"https://image.tmdb.org/t/p/w500{poster}" if poster else ""
            overview = movie.get('overview', "No description.")
            return poster_url, overview
    except:
        pass

    return "", "Description not available."

# ===============================
# RECOMMENDATION MODELS
# ===============================
def get_collaborative_recs(movie_name):
    user_movie_rating = movie_matrix[movie_name]
    similar_movies = movie_matrix.corrwith(user_movie_rating)
    corr_movie = similar_movies.to_frame(name='CF_Score')
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings_count)
    recs = corr_movie[corr_movie['num_of_ratings'] > 50]
    recs['CF_Score'] = ((recs['CF_Score'] + 1) / 2) * 100
    return recs.sort_values('CF_Score', ascending=False)


def get_content_based_recs(movie_name):
    cosine_sim = cosine_similarity(genre_data)
    sim_df = pd.DataFrame(cosine_sim,
                          index=genre_data.index,
                          columns=genre_data.index)
    movie_scores = sim_df[movie_name] * 100
    return movie_scores.to_frame(name='CB_Score').sort_values('CB_Score', ascending=False)


def get_hybrid_recs(movie_name):
    cf = get_collaborative_recs(movie_name)
    cb = get_content_based_recs(movie_name)
    hybrid = cf.join(cb, how='inner')
    hybrid['Hybrid_Score'] = (hybrid['CF_Score'] * 0.5) + (hybrid['CB_Score'] * 0.5)
    return hybrid.sort_values('Hybrid_Score', ascending=False)


def filter_by_genre(recs, selected_genre):
    if selected_genre == "All Genres":
        return recs
    valid = genre_data[genre_data[selected_genre] == 1].index
    return recs[recs.index.isin(valid)]

# ===============================
# HEADER
# ===============================
st.markdown('<div class="main-title">CineMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Movie Intelligence System</div>', unsafe_allow_html=True)

# ===============================
# CONTROLS
# ===============================
col1, col2, col3 = st.columns(3)

selected_movie = col1.selectbox("🎬 Select Movie", movie_list)
selected_genre = col2.selectbox("🎭 Filter Genre", genre_list)
model_type = col3.selectbox("🧠 Model",
                            ["Hybrid", "Collaborative", "Content-Based"])

# ===============================
# TABS
# ===============================
tab1, tab2 = st.tabs(["🎬 Recommendations", "📊 Dataset Analytics"])

# ===============================
# TAB 1 – RECOMMENDATIONS
# ===============================
with tab1:

    st.markdown('<div class="section-header">Top Recommendations</div>',
                unsafe_allow_html=True)

    if model_type == "Hybrid":
        recs = get_hybrid_recs(selected_movie)
        score_col = "Hybrid_Score"
    elif model_type == "Collaborative":
        recs = get_collaborative_recs(selected_movie)
        score_col = "CF_Score"
    else:
        recs = get_content_based_recs(selected_movie)
        score_col = "CB_Score"

    recs = filter_by_genre(recs, selected_genre).head(6)

    cols = st.columns(3)

    for i, (movie, row) in enumerate(recs.iterrows()):
        poster, overview = fetch_movie_details(movie)

        with cols[i % 3]:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            if poster:
                st.image(poster, use_container_width=True)

            st.markdown(f"### {clean_movie_title(movie)}")
            st.markdown(
                f'<div class="score-badge">{row[score_col]:.0f}% Match</div>',
                unsafe_allow_html=True
            )
            st.write(overview[:150] + "...")

            st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# TAB 2 – ANALYTICS
# ===============================
with tab2:

    st.markdown('<div class="section-header">Dataset Insights</div>',
                unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### 🎭 Genre Distribution")
        genre_counts = genre_data.sum().sort_values(ascending=False)
        st.bar_chart(genre_counts)

    with colB:
        st.markdown("### ⭐ Rating Distribution")
        rating_counts = ratings['rating'].value_counts().sort_index()
        st.bar_chart(rating_counts)

    st.markdown("### 🎬 Top 10 Most Rated Movies")
    top_movies = ratings_count.sort_values(
        'num_of_ratings', ascending=False).head(10)
    st.bar_chart(top_movies['num_of_ratings'])
