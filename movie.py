import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🍿", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 3.5rem; color: #E50914; text-align: center; font-weight: 900; margin-bottom: 0px; }
    .sub-title { text-align: center; color: #b3b3b3; font-size: 1.2rem; margin-bottom: 2rem; }
    .movie-card { background-color: #1e1e1e; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3); height: 100%; border: 1px solid #333; }
    .match-score { color: #46d369; font-weight: bold; font-size: 1.2rem; }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING & CACHING ---
@st.cache_data
def load_and_prep_data():
    # Load Ratings
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('ml-100k/u.data', sep='\t', names=column_names)
    
    # Load Movies and Genres (MovieLens has 19 genre columns)
    movie_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 
                  'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 
                  'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                  'Sci-Fi', 'Thriller', 'War', 'Western']
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, names=movie_cols)
    
    # Merge for Collaborative Filtering
    merged_df = pd.merge(df, movies[['item_id', 'title']], on='item_id')
    movie_matrix = merged_df.pivot_table(index='user_id', columns='title', values='rating')
    
    ratings_count = pd.DataFrame(merged_df.groupby('title')['rating'].count())
    ratings_count.rename(columns={'rating': 'num_of_ratings'}, inplace=True)
    
    # Prepare data for Content-Based Filtering (Isolate the genres)
    genre_data = movies.drop(columns=['item_id', 'release_date', 'video_release_date', 'imdb_url', 'unknown'])
    genre_data = genre_data.groupby('title').max() # Clean duplicate titles
    
    return movie_matrix, ratings_count, genre_data, sorted(movies['title'].unique())

movie_matrix, ratings_count, genre_data, movie_list = load_and_prep_data()

# --- ALGORITHM 1: COLLABORATIVE FILTERING ---
def get_collaborative_recs(movie_name, min_reviews=50):
    user_movie_rating = movie_matrix[movie_name]
    similar_movies = movie_matrix.corrwith(user_movie_rating)
    
    corr_movie = pd.DataFrame(similar_movies, columns=['CF_Score'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings_count['num_of_ratings'])
    
    # Filter and normalize score to 0-100%
    recs = corr_movie[corr_movie['num_of_ratings'] > min_reviews].copy()
    recs['CF_Score'] = ((recs['CF_Score'] + 1) / 2) * 100 # Convert -1 to 1 range to 0-100%
    return recs.sort_values('CF_Score', ascending=False).drop(movie_name, errors='ignore')

# --- ALGORITHM 2: CONTENT-BASED FILTERING ---
def get_content_based_recs(movie_name):
    # Calculate cosine similarity between all movies based on their 19 genre tags
    cosine_sim = cosine_similarity(genre_data)
    sim_df = pd.DataFrame(cosine_sim, index=genre_data.index, columns=genre_data.index)
    
    # Get scores for the specific movie and convert to 0-100%
    movie_scores = sim_df[movie_name] * 100 
    recs = pd.DataFrame(movie_scores, columns=['CB_Score'])
    return recs.sort_values('CB_Score', ascending=False).drop(movie_name, errors='ignore')

# --- ALGORITHM 3: HYBRID MODEL ---
def get_hybrid_recs(movie_name, min_reviews=50):
    cf_recs = get_collaborative_recs(movie_name, min_reviews)
    cb_recs = get_content_based_recs(movie_name)
    
    # Combine both dataframes
    hybrid_df = cf_recs.join(cb_recs, how='inner')
    
    # Hybrid calculation: 50% Collaborative Score + 50% Content-Based Score
    hybrid_df['Hybrid_Score'] = (hybrid_df['CF_Score'] * 0.5) + (hybrid_df['CB_Score'] * 0.5)
    return hybrid_df.sort_values('Hybrid_Score', ascending=False)

# --- HELPER FUNCTION TO RENDER UI CARDS ---
def render_movie_cards(recommendations, score_column):
    cols = st.columns(5)
    for i, (index, row) in enumerate(recommendations.head(5).iterrows()):
        with cols[i]:
            st.markdown(f'''
                <div class="movie-card">
                    <h4>🎬 {index}</h4>
                    <p class="match-score">{row[score_column]:.0f}% Match</p>
                </div>
            ''', unsafe_allow_html=True)

# --- UI LAYOUT ---
st.markdown('<p class="main-title">🍿 CineMatch</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Your Personalized Movie Recommender System</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["👥 Collaborative Filtering", "📖 Content-Based Filtering", "🤖 Hybrid Model"])

with tab1:
    st.markdown("### Item-Based Collaborative Filtering")
    st.write("Find movies similar to your favorites based on community rating patterns.")
    col1, col2 = st.columns([4, 1])
    selected_movie_cf = col1.selectbox("Search for a movie:", movie_list, key="cf")
    if col2.button("Recommend", key="btn_cf", use_container_width=True, type="primary"):
        with st.spinner('Calculating user correlations...'):
            try:
                recs = get_collaborative_recs(selected_movie_cf)
                render_movie_cards(recs, 'CF_Score')
            except:
                st.error("Not enough rating data for this movie. Try a more popular title.")

with tab2:
    st.markdown("### Content-Based Filtering")
    st.write("Find movies with similar genres, ignoring user ratings completely.")
    col1, col2 = st.columns([4, 1])
    selected_movie_cb = col1.selectbox("Search for a movie:", movie_list, key="cb")
    if col2.button("Recommend", key="btn_cb", use_container_width=True, type="primary"):
        with st.spinner('Calculating genre cosine similarities...'):
            try:
                recs = get_content_based_recs(selected_movie_cb)
                render_movie_cards(recs, 'CB_Score')
            except Exception as e:
                st.error(f"Error: {e}")

with tab3:
    st.markdown("### Hybrid Model")
    st.write("The ultimate recommender: Combines 50% Collaborative logic and 50% Content-Based logic.")
    col1, col2 = st.columns([4, 1])
    selected_movie_hy = col1.selectbox("Search for a movie:", movie_list, key="hy")
    if col2.button("Recommend", key="btn_hy", use_container_width=True, type="primary"):
        with st.spinner('Fusing algorithms...'):
            try:
                recs = get_hybrid_recs(selected_movie_hy)
                render_movie_cards(recs, 'Hybrid_Score')
            except:
                st.error("Not enough data to create a hybrid score for this movie. Try a more popular title.")
