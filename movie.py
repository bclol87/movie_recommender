import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity

# --- TMDB API CONFIGURATION ---
API_KEY = "3eb39709869b67fd15b086e095c5cbec"

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="CineMatch", page_icon="🍿", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 4rem; color: #E50914; font-weight: 900; margin-bottom: 0px; letter-spacing: -1px; }
    .sub-title { color: #aaaaaa; font-size: 1.2rem; margin-bottom: 2rem; font-weight: 400; }
    .category-header { font-size: 1.5rem; color: #ffffff; font-weight: bold; margin-top: 2rem; margin-bottom: 1rem; border-left: 5px solid #E50914; padding-left: 10px; }
    
    /* Upgraded Card for Posters and Descriptions */
    .movie-card { 
        background: #181818;
        padding: 15px; 
        border-radius: 10px; 
        text-align: center; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.5); 
        height: 100%; 
        display: flex;
        flex-direction: column;
        border: 1px solid #333;
        transition: transform 0.2s;
    }
    .movie-card:hover { transform: scale(1.03); border-color: #E50914; }
    
    .movie-poster { width: 100%; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.5); }
    .movie-title { font-size: 1.1rem; color: white; font-weight: bold; margin-bottom: 5px; line-height: 1.2; }
    .match-score { color: #46d369; font-weight: bold; font-size: 1rem; margin-bottom: 10px;}
    .movie-overview { font-size: 0.8rem; color: #bbbbbb; text-align: left; margin-bottom: 15px; flex-grow: 1; line-height: 1.4; }
    
    .watch-btn { 
        background-color: #E50914; color: white !important; padding: 8px; 
        border-radius: 4px; text-decoration: none; font-weight: bold; 
        display: block; width: 100%; margin-top: auto; 
    }
    .watch-btn:hover { background-color: #f40612; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING & CACHING ---
@st.cache_data
def load_and_prep_data():
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('ml-100k/u.data', sep='\t', names=column_names)
    
    movie_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 
                  'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 
                  'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                  'Sci-Fi', 'Thriller', 'War', 'Western']
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, names=movie_cols)
    
    merged_df = pd.merge(df, movies[['item_id', 'title']], on='item_id')
    movie_matrix = merged_df.pivot_table(index='user_id', columns='title', values='rating')
    
    ratings_count = pd.DataFrame(merged_df.groupby('title')['rating'].count())
    ratings_count.rename(columns={'rating': 'num_of_ratings'}, inplace=True)
    
    genre_data = movies.drop(columns=['item_id', 'release_date', 'video_release_date', 'imdb_url', 'unknown'])
    genre_data = genre_data.groupby('title').max()
    
    return movie_matrix, ratings_count, genre_data, sorted(movies['title'].unique())

movie_matrix, ratings_count, genre_data, movie_list = load_and_prep_data()

# --- API HELPER FUNCTIONS ---
def clean_movie_title(title):
    # Removes the year e.g. " (1995)"
    clean_title = re.sub(r'\s*\(\d{4}\)', '', title).strip()
    # Fixes grammatical shifting e.g. "Matrix, The" -> "The Matrix"
    if clean_title.endswith(', The'):
        clean_title = 'The ' + clean_title[:-5]
    elif clean_title.endswith(', A'):
        clean_title = 'A ' + clean_title[:-3]
    return clean_title

@st.cache_data(ttl=86400) # Cache API calls for 24 hours so it stays fast
def fetch_movie_details(movie_title):
    search_title = clean_movie_title(movie_title)
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={search_title}"
    
    try:
        response = requests.get(url)
        data = response.json()
        if data['results']:
            movie = data['results'][0]
            
            # 1. Poster URL
            poster_path = movie.get('poster_path')
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/500x750?text=No+Poster"
            
            # 2. Overview (Truncate if too long)
            overview = movie.get('overview', 'No description available.')
            if len(overview) > 120:
                overview = overview[:117] + "..."
                
            # 3. TMDB Movie Link
            movie_id = movie.get('id')
            movie_link = f"https://www.themoviedb.org/movie/{movie_id}"
            
            return poster_url, overview, movie_link
    except:
        pass
    
    # Fallback if API fails or movie isn't found
    return "https://via.placeholder.com/500x750?text=No+Poster", "Description not found.", "#"

# --- CORE ALGORITHMS ---
def get_collaborative_recs(movie_name, min_reviews=50):
    user_movie_rating = movie_matrix[movie_name]
    similar_movies = movie_matrix.corrwith(user_movie_rating)
    corr_movie = similar_movies.to_frame(name='CF_Score')
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings_count['num_of_ratings'])
    recs = corr_movie[corr_movie['num_of_ratings'] > min_reviews].copy()
    recs['CF_Score'] = ((recs['CF_Score'] + 1) / 2) * 100 
    return recs.sort_values('CF_Score', ascending=False).drop(movie_name, errors='ignore')

def get_content_based_recs(movie_name):
    cosine_sim = cosine_similarity(genre_data)
    sim_df = pd.DataFrame(cosine_sim, index=genre_data.index, columns=genre_data.index)
    movie_scores = sim_df[movie_name] * 100 
    recs = movie_scores.to_frame(name='CB_Score')
    return recs.sort_values('CF_Score', ascending=False).drop(movie_name, errors='ignore')

def get_hybrid_recs(movie_name, min_reviews=50):
    cf_recs = get_collaborative_recs(movie_name, min_reviews)
    cb_recs = get_content_based_recs(movie_name)
    hybrid_df = cf_recs.join(cb_recs, how='inner')
    hybrid_df['Hybrid_Score'] = (hybrid_df['CF_Score'] * 0.5) + (hybrid_df['CB_Score'] * 0.5)
    return hybrid_df.sort_values('Hybrid_Score', ascending=False)

# --- HELPER FUNCTION TO RENDER UI CARDS ---
def render_movie_cards(recommendations, score_column, model_type):
    cols = st.columns(5)
    for i, (index, row) in enumerate(recommendations.head(5).iterrows()):
        
        # Fetch live data from TMDB API
        poster_url, overview, movie_link = fetch_movie_details(index)

        with cols[i]:
            st.markdown(f'''
                <div class="movie-card">
                    <img src="{poster_url}" class="movie-poster" alt="{index}">
                    <div class="movie-title">{clean_movie_title(index)}</div>
                    <div class="match-score">{row[score_column]:.0f}% Match</div>
                    <div class="movie-overview">{overview}</div>
                    <a href="{movie_link}" target="_blank" class="watch-btn">View Details</a>
                </div>
            ''', unsafe_allow_html=True)

# --- MAIN UI LAYOUT ---
st.markdown('<p class="main-title">CineMatch</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Movies, shows, and more. Tailored to you.</p>', unsafe_allow_html=True)

# Search Bar Area
col1, col2 = st.columns([4, 1])
selected_movie = col1.selectbox("Search for a movie:", movie_list, label_visibility="collapsed")
generate_btn = col2.button("Find Movies", type="primary", use_container_width=True)

if generate_btn:
    with st.spinner('Fetching movie posters and curating your dashboard...'):
        
        # ROW 1: Hybrid Model
        st.markdown('<p class="category-header">✨ Top Picks For You</p>', unsafe_allow_html=True)
        try:
            hy_recs = get_hybrid_recs(selected_movie)
            render_movie_cards(hy_recs, 'Hybrid_Score', 'Hybrid')
        except Exception as e:
            st.error("Could not calculate Top Picks for this movie.")
            
        # ROW 2: Collaborative Model
        st.markdown('<p class="category-header">👥 What The Community Is Watching</p>', unsafe_allow_html=True)
        try:
            cf_recs = get_collaborative_recs(selected_movie)
            render_movie_cards(cf_recs, 'CF_Score', 'CF')
        except Exception as e:
            st.error("Not enough community data for this title.")
        
        # ROW 3: Content-Based
        st.markdown('<p class="category-header">🎭 Similar Vibe & Genres</p>', unsafe_allow_html=True)
        try:
            cb_recs = get_content_based_recs(selected_movie)
            render_movie_cards(cb_recs, 'CB_Score', 'CB')
        except Exception as e:
            st.error("Error generating genre recommendations.")
