import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import difflib
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- TMDB API CONFIGURATION ---
# We will move this to Streamlit Secrets next!
API_KEY = "3eb39709869b67fd15b086e095c5cbec"

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="CineMatch Pro", page_icon="🍿", layout="wide")

st.markdown("""
    <style>
    /* Main Title with a fiery gradient */
    .main-title { 
        font-size: 4rem; 
        font-weight: 900; 
        margin-bottom: 0px; 
        text-align: center;
        background: linear-gradient(90deg, #E50914, #ff7b00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title { 
        color: #aaaaaa; 
        font-size: 1.2rem; 
        margin-bottom: 2rem; 
        font-weight: 400; 
        text-align: center;
    }
    
    /* Modern Category Headers */
    .category-header { 
        font-size: 1.5rem; 
        color: #000000 !important; /* <--- THIS FORCES IT TO BE BLACK */
        font-weight: bold; 
        margin-top: 2rem; 
        margin-bottom: 1rem; 
        border-left: 5px solid #E50914; 
        padding-left: 10px; 
    }
    
    /* Premium Glassmorphism Movie Cards */
    .movie-card { 
        background: #222222; 
        padding: 15px; 
        border-radius: 10px; 
        text-align: center; 
        height: 600px; /* 1. STRICT HEIGHT: Forces all boxes to be exactly the same size */
        display: flex;
        flex-direction: column; 
        border: 1px solid #333333; 
        transition: transform 0.2s, border-color 0.2s;
    }
    .movie-card:hover { 
        transform: translateY(-5px); 
        border-color: #E50914; 
    }
    
    .movie-poster { 
        width: 100%; 
        height: 320px; /* 2. STRICT POSTER HEIGHT */
        object-fit: cover; 
        border-radius: 8px; 
        margin-bottom: 12px; 
    }
    
    .movie-title { 
        font-size: 1.1rem; 
        color: white; 
        font-weight: bold; 
        margin-bottom: 5px; 
        min-height: 2.8rem; /* 3. LOCK TITLE HEIGHT: Allows 2 lines of text without shifting */
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;  
        overflow: hidden;
    }
    
    .match-score { 
        color: #46d369; 
        font-weight: bold; 
        font-size: 1rem; 
        margin-bottom: 10px; 
        min-height: 1.2rem; /* Ensures space is kept even if the score is missing */
    }
    
    .movie-overview { 
        font-size: 0.85rem; 
        color: #bbbbbb; 
        text-align: left; 
        margin-bottom: 15px; 
        line-height: 1.4;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;  
        overflow: hidden;
        flex-grow: 1; /* 4. FILLS EMPTY SPACE */
    }
    
    .watch-btn { 
        background-color: #E50914; 
        color: white !important; 
        padding: 8px; 
        border-radius: 4px; 
        text-decoration: none; 
        font-weight: bold; 
        display: block; 
        width: 100%; 
        margin-top: auto; /* 5. PUSHES BUTTON TO THE VERY BOTTOM ALIGNMENT */
    }
    .watch-btn:hover { background-color: #f40612; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING & AI TRAINING ---
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('movies_final_lite.csv')
    
    # 1. Parse the JSON genres into a clean string
    def extract_genres(x):
        try:
            genres = ast.literal_eval(x)
            return " ".join([g['name'] for g in genres])
        except:
            return ""
            
    df['genres_clean'] = df['genres'].apply(extract_genres)
    
    # 2. Clean missing data
    df['overview'] = df['overview'].fillna('')
    df['actors'] = df['actors'].fillna('')
    df['director'] = df['director'].fillna('')
    df['title'] = df['title'].astype(str)
    
    # 3. Create the "Content DNA" for the AI
    df['content_features'] = df['genres_clean'] + " " + df['actors'] + " " + df['director'] + " " + df['overview']
    
    return df.dropna(subset=['title']).reset_index(drop=True)

@st.cache_resource
def train_tfidf_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

movies = load_and_prep_data()
cosine_sim = train_tfidf_model(movies)
movie_list = sorted(movies['title'].unique())

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
        for movie in data.get('results', [])[:5]:
            p_url, desc, link = fetch_movie_details(movie['id'], is_id=True)
            results.append({'title': movie.get('title'), 'poster': p_url, 'overview': desc, 'link': link, 'score': movie.get('vote_average', 0) * 10})
        return results
    except: return []

# --- CORE ALGORITHMS ---
def get_content_based_recs(movie_title):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Grab the top 20 instead of the top 5!
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:21]
    
    recs = movies.iloc[[i[0] for i in sim_scores]].copy()
    recs['CB_Score'] = [i[1] * 100 for i in sim_scores]
    return recs

def get_community_recs(movie_title):
    idx = movies[movies['title'] == movie_title].index[0]
    target_genres = movies.iloc[idx]['genres_clean'].split()
    
    if not target_genres: return movies.head(20)
    
    pattern = '|'.join(target_genres)
    pool = movies[movies['genres_clean'].str.contains(pattern, case=False, na=False)].copy()
    pool = pool[pool['title'] != movie_title]
    
    pool['CF_Score'] = (pool['vote_average'] / 10) * 100
    # Sort and return the top 20
    return pool.sort_values(['vote_count', 'vote_average'], ascending=[False, False]).head(20)

def get_hybrid_recs(movie_title):
    cb = get_content_based_recs(movie_title)
    cf = get_community_recs(movie_title)
    
    hybrid = pd.concat([cb, cf]).drop_duplicates(subset=['id'])
    hybrid['Hybrid_Score'] = ((hybrid['vote_average']/10)*100 * 0.3) + (hybrid.get('CB_Score', 50) * 0.7)
    # Sort and return the top 20
    return hybrid.sort_values('Hybrid_Score', ascending=False).head(20)

# --- HELPER FUNCTION TO RENDER UI CARDS ---
def render_movie_cards(recommendations, score_column, category_key):
    top_5 = recommendations.head(5)
    
    # Create 6 columns (5 for movies, 1 slightly thinner one for the button)
    cols = st.columns([1, 1, 1, 1, 1, 0.7]) 
    
    for i, (_, row) in enumerate(top_5.iterrows()):
        poster_url, overview, movie_link = fetch_movie_details(row['title'])
        with cols[i]:
            st.markdown(f'''
                <div class="movie-card">
                    <img src="{poster_url}" class="movie-poster" alt="poster">
                    <div class="movie-title">{row['title']}</div>
                    <div class="match-score">{row.get(score_column, 85):.0f}% Match</div>
                    <div class="movie-overview">{overview}</div>
                    <a href="{movie_link}" target="_blank" class="watch-btn">View Details</a>
                </div>
            ''', unsafe_allow_html=True)
            
    # The 6th Column for the "Show More" Button
    with cols[5]:
        # Add vertical space to center the button next to the tall movie cards
        st.markdown('<div style="height: 180px;"></div>', unsafe_allow_html=True)
        
        # Setup the toggle memory so Streamlit remembers if it's open or closed
        state_key = f"show_{category_key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = False
            
        # Draw the interactive button
        if st.button("➕ Show More" if not st.session_state[state_key] else "➖ Show Less", key=f"btn_{category_key}", use_container_width=True):
            st.session_state[state_key] = not st.session_state[state_key]
            st.rerun() # Refresh screen smoothly

    # If the user clicked "Show More", reveal the next 15 movies below!
    if st.session_state.get(state_key, False) and len(recommendations) > 5:
        st.markdown("---") 
        remaining = recommendations.iloc[5:]
        
        # Draw the extra movies in perfect rows of 5
        for row_start in range(0, len(remaining), 5):
            chunk = remaining.iloc[row_start:row_start+5]
            exp_cols = st.columns(5) 
            
            for j, (_, row) in enumerate(chunk.iterrows()):
                poster_url, overview, movie_link = fetch_movie_details(row['title'])
                with exp_cols[j]:
                    st.markdown(f'''
                        <div class="movie-card">
                            <img src="{poster_url}" class="movie-poster">
                            <div class="movie-title">{row['title']}</div>
                            <div class="match-score">{row.get(score_column, 85):.0f}% Match</div>
                            <div class="movie-overview">{overview}</div>
                            <a href="{movie_link}" target="_blank" class="watch-btn">View Details</a>
                        </div>
                    ''', unsafe_allow_html=True)

# --- MAIN UI LAYOUT ---
st.markdown('<p class="main-title">CineMatch Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by Content-Based & Community Algorithms.</p>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
search_query = col1.text_input("Search", placeholder="Type a movie title or topic (e.g., 'car', 'space')...", label_visibility="collapsed")
selected_display = col2.selectbox("Choose Model:", ["Show All Rows", "✨ Top Picks (Hybrid)", "👥 Community Picks", "🎭 AI Similar (Content-Based)"], label_visibility="collapsed")

st.divider()

if search_query:
    with st.spinner('Curating dashboard...'):
        closest_matches = difflib.get_close_matches(search_query.title(), movie_list, n=1, cutoff=0.5)
        
        if closest_matches:
            selected_movie = closest_matches[0]

            idx = movies[movies['title'] == selected_movie].index[0]
            movie_type = movies.iloc[idx]['genres_clean'].replace(" ", ", ")
            
            st.success(f"🎯 Local AI Models activated for: **{selected_movie}**")

            st.info(f"🏷️ **Movie Type:** {movie_type}")
            
            if selected_display in ["Show All Rows", "✨ Top Picks (Hybrid)"]:
                st.markdown('<p class="category-header">✨ Hybrid Top Picks</p>', unsafe_allow_html=True)
                # ADDED 'hybrid' AT THE END
                render_movie_cards(get_hybrid_recs(selected_movie), 'Hybrid_Score', 'hybrid')
                    
            if selected_display in ["Show All Rows", "👥 Community Picks"]:
                st.markdown('<p class="category-header">👥 Community Favorites</p>', unsafe_allow_html=True)
                # ADDED 'community' AT THE END
                render_movie_cards(get_community_recs(selected_movie), 'CF_Score', 'community')
            
            if selected_display in ["Show All Rows", "🎭 AI Similar (Content-Based)"]:
                st.markdown('<p class="category-header">🎭 Content Similarity</p>', unsafe_allow_html=True)
                # ADDED 'content' AT THE END
                render_movie_cards(get_content_based_recs(selected_movie), 'CB_Score', 'content')
                    
        else:
            st.info(f"🌐 Searching global TMDB database for topic: **'{search_query}'**")
            topic_results = search_tmdb_topic(search_query)
            
            if topic_results:
                cols = st.columns(len(topic_results))
                for i, movie in enumerate(topic_results):
                    with cols[i]:
                        st.markdown(f'''
                            <div class="movie-card">
                                <img src="{movie['poster']}" class="movie-poster">
                                <div class="movie-title">{movie['title']}</div>
                                <div class="match-score">{movie['score']:.0f}% TMDB Score</div>
                                <div class="movie-overview">{movie['overview']}</div>
                                <a href="{movie['link']}" target="_blank" class="watch-btn">View Details</a>
                            </div>
                        ''', unsafe_allow_html=True)
            else:
                st.warning("No movies found for that topic.")
else:
    st.info("👆 Type a movie name (e.g., 'Iron Man') to run your AI models, or a topic (e.g., 'car') to search globally!")
