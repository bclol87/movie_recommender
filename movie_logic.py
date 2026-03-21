import streamlit as st
import pandas as pd
import requests
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- TMDB API CONFIGURATION ---
API_KEY = "3eb39709869b67fd15b086e095c5cbec"

# --- DATA LOADING & AI TRAINING ---
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('movies_final_lite.csv')
    
    def extract_genres(x):
        try:
            genres = ast.literal_eval(x)
            return " ".join([g['name'] for g in genres])
        except:
            return ""
            
    df['genres_clean'] = df['genres'].apply(extract_genres)
    df['overview'] = df['overview'].fillna('')
    df['actors'] = df['actors'].fillna('')
    df['director'] = df['director'].fillna('')
    df['title'] = df['title'].astype(str)
    
    # NLP UPGRADE
    df['content_features'] = df['title'] + " " + df['title'] + " " + df['genres_clean'] + " " + df['actors'] + " " + df['director'] + " " + df['overview']
    return df.dropna(subset=['title']).reset_index(drop=True)

@st.cache_resource
def train_tfidf_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim

movies = load_and_prep_data()
tfidf, tfidf_matrix, cosine_sim = train_tfidf_model(movies)

# --- API HELPER FUNCTIONS ---
@st.cache_data(ttl=86400) 
def fetch_movie_details(title_or_id, is_id=False):
    query_param = f"/{title_or_id}" if is_id else f"/search/movie?query={title_or_id}"
    url = f"https://api.themoviedb.org/3{query_param}&api_key={API_KEY}" if not is_id else f"https://api.themoviedb.org/3/movie/{title_or_id}?api_key={API_KEY}"
    
    try:
        response = requests.get(url).json()
        movie = response if is_id else response['results'][0]
        
        poster_path = movie.get('poster_path')
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/500x750?text=No+Poster"
        
        overview = movie.get('overview', 'No description available.')
        if len(overview) > 120: overview = overview[:117] + "..."
            
        return poster_url, overview, f"https://www.themoviedb.org/movie/{movie.get('id')}"
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster", "Description not found.", "#"

def search_tmdb_topic(query):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"
    try:
        data = requests.get(url).json()
        results = []
        for movie in data.get('results', [])[:20]:
            p_url, desc, link = fetch_movie_details(movie['id'], is_id=True)
            results.append({'title': movie.get('title'), 'poster': p_url, 'overview': desc, 'link': link, 'score': movie.get('vote_average', 0) * 10})
        return results
    except: return []

# --- CORE ALGORITHMS ---
def get_content_based_recs(movie_title):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[0:20]
    
    recs = movies.iloc[[i[0] for i in sim_scores]].copy()
    recs['CB_Score'] = [i[1] * 100 for i in sim_scores]
    return recs

def get_community_recs(movie_title):
    idx = movies[movies['title'] == movie_title].index[0]
    target_genres = movies.iloc[idx]['genres_clean'].split()
    if not target_genres: return movies.head(20)
    
    pattern = '|'.join(target_genres)
    pool = movies[movies['genres_clean'].str.contains(pattern, case=False, na=False)].copy()
    
    pool['CF_Score'] = (pool['vote_average'] / 10) * 100
    return pool.sort_values(['vote_count', 'vote_average'], ascending=[False, False]).head(20)

def get_hybrid_recs(movie_title):
    cb = get_content_based_recs(movie_title)
    cf = get_community_recs(movie_title)
    
    hybrid = pd.concat([cb, cf]).drop_duplicates(subset=['id'])
    hybrid['Hybrid_Score'] = ((hybrid['vote_average']/10)*100 * 0.3) + (hybrid.get('CB_Score', 50) * 0.7)
    return hybrid.sort_values('Hybrid_Score', ascending=False).head(20)

def get_profile_based_recs(liked_titles):
    """Generates recommendations based on an array of liked movie titles."""
    if not liked_titles:
        return pd.DataFrame()
    
    # Find the indices of the movies the user has liked
    indices = movies[movies['title'].isin(liked_titles)].index
    if len(indices) == 0:
        return pd.DataFrame()
    
    # MAGIC: Combine the TF-IDF vectors of all liked movies into one "User Profile"
    user_profile = np.asarray(tfidf_matrix[indices].mean(axis=0))
    
    # Calculate similarity of ALL movies against the combined user profile
    sim_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
    
    # Sort and grab top matches, excluding the ones they already liked
    sim_scores_indices = sim_scores.argsort()[::-1]
    top_indices = [i for i in sim_scores_indices if movies.iloc[i]['title'] not in liked_titles][:20]
    
    recs = movies.iloc[top_indices].copy()
    recs['Profile_Score'] = sim_scores[top_indices] * 100
    return recs
