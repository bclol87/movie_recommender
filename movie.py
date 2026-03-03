import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity

# --- TMDB API CONFIGURATION ---
API_KEY = "3eb39709869b67fd15b086e095c5cbec"

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CineMatch", page_icon="🍿", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-title { font-size: 4rem; color: #E50914; font-weight: 900; }
    .sub-title { color: #555555; font-size: 1.2rem; margin-bottom: 2rem; }
    .category-header { font-size: 1.5rem; font-weight: bold; margin-top: 2rem; }
    .movie-card { 
        background: #181818; padding: 15px; border-radius: 10px; 
        text-align: center; color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_and_prep_data():
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=column_names)

    movie_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 
                  'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 
                  'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                  'Sci-Fi', 'Thriller', 'War', 'Western']

    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                         header=None, names=movie_cols)

    merged_df = pd.merge(ratings, movies[['item_id', 'title']], on='item_id')
    movie_matrix = merged_df.pivot_table(index='user_id', columns='title', values='rating')

    ratings_count = pd.DataFrame(merged_df.groupby('title')['rating'].count())
    ratings_count.rename(columns={'rating': 'num_of_ratings'}, inplace=True)

    genre_data = movies.drop(columns=['item_id', 'release_date',
                                      'video_release_date', 'imdb_url', 'unknown'])
    genre_data = genre_data.groupby('title').max().fillna(0).astype(int)

    genre_list = ['All Genres'] + list(genre_data.columns)

    return ratings, movie_matrix, ratings_count, genre_data, sorted(movies['title'].unique()), genre_list


ratings, movie_matrix, ratings_count, genre_data, movie_list, genre_list = load_and_prep_data()

# --- API HELPER ---
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


# --- RECOMMENDATION MODELS ---
def get_collaborative_recs(movie_name):
    user_movie_rating = movie_matrix[movie_name]
    similar_movies = movie_matrix.corrwith(user_movie_rating)
    corr_movie = similar_movies.to_frame(name='CF_Score')
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings_count['num_of_ratings'])
    recs = corr_movie[corr_movie['num_of_ratings'] > 50]
    recs['CF_Score'] = ((recs['CF_Score'] + 1) / 2) * 100
    return recs.sort_values('CF_Score', ascending=False)


def get_content_based_recs(movie_name):
    cosine_sim = cosine_similarity(genre_data)
    sim_df = pd.DataFrame(cosine_sim,
                          index=genre_data.index,
                          columns=genre_data.index)
    movie_scores = sim_df[movie_name] * 100
    recs = movie_scores.to_frame(name='CB_Score')
    return recs.sort_values('CB_Score', ascending=False)


def get_hybrid_recs(movie_name):
    cf = get_collaborative_recs(movie_name)
    cb = get_content_based_recs(movie_name)
    hybrid = cf.join(cb, how='inner')
    hybrid['Hybrid_Score'] = (hybrid['CF_Score'] * 0.5) + (hybrid['CB_Score'] * 0.5)
    return hybrid.sort_values('Hybrid_Score', ascending=False)


def filter_by_genre(recommendations, selected_genre):
    if selected_genre == 'All Genres':
        return recommendations
    valid_movies = genre_data[genre_data[selected_genre] == 1].index
    return recommendations[recommendations.index.isin(valid_movies)]


# --- UI ---
st.markdown('<p class="main-title">CineMatch</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Tailored Movie Recommendations</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

selected_movie = col1.selectbox("Select Movie", movie_list)
selected_genre = col2.selectbox("Filter Genre", genre_list)
model_type = col3.selectbox("Choose Model",
                            ["Hybrid", "Collaborative", "Content-Based"])

st.divider()

with st.spinner("Generating Recommendations..."):

    if model_type == "Hybrid":
        recs = get_hybrid_recs(selected_movie)
        recs = filter_by_genre(recs, selected_genre)
        recs = recs.head(5)
        score_col = "Hybrid_Score"

    elif model_type == "Collaborative":
        recs = get_collaborative_recs(selected_movie)
        recs = filter_by_genre(recs, selected_genre)
        recs = recs.head(5)
        score_col = "CF_Score"

    else:
        recs = get_content_based_recs(selected_movie)
        recs = filter_by_genre(recs, selected_genre)
        recs = recs.head(5)
        score_col = "CB_Score"

    for movie, row in recs.iterrows():
        poster, overview = fetch_movie_details(movie)
        st.markdown(f"### {clean_movie_title(movie)}")
        if poster:
            st.image(poster, width=200)
        st.write(f"Match Score: {row[score_col]:.2f}%")
        st.write(overview)
        st.divider()

# ==================================================
# 📊 DATASET ANALYTICS SECTION
# ==================================================

st.markdown("## 📊 Dataset Insights")

with st.expander("View Dataset Analytics"):

    # 1️⃣ Top Genres Distribution
    st.markdown("### 🎭 Top Genres Distribution")
    genre_counts = genre_data.sum().sort_values(ascending=False)
    st.bar_chart(genre_counts)

    # 2️⃣ Rating Distribution
    st.markdown("### ⭐ Rating Distribution")
    rating_counts = ratings['rating'].value_counts().sort_index()
    st.bar_chart(rating_counts)

    # 3️⃣ Top 10 Most Rated Movies
    st.markdown("### 🎬 Top 10 Most Rated Movies")
    top_movies = ratings_count.sort_values('num_of_ratings',
                                           ascending=False).head(10)
    st.bar_chart(top_movies['num_of_ratings'])
