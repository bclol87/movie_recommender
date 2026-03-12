import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import base64

# --- MAGIC STEP: Import our functions from our new logic file ---
from movie_logic import (
    movies, cosine_sim, tfidf, tfidf_matrix,
    fetch_movie_details, search_tmdb_topic,
    get_content_based_recs, get_community_recs, get_hybrid_recs
)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CineMatch Pro", page_icon="🎬", layout="wide")

# --- NETFLIX-INSPIRED CSS ---
st.markdown("""
    <style>
        /* Main background - Netflix black */
        .stApp {
            background-color: #141414 !important;
            color: #ffffff !important;
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #333;
        }
        ::-webkit-scrollbar-thumb {
            background: #666;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #e50914;
        }
        
        /* Netflix logo style */
        .netflix-logo {
            font-size: 2.5rem;
            font-weight: 800;
            color: #e50914;
            text-transform: uppercase;
            letter-spacing: -1px;
            margin-bottom: 0;
            line-height: 1;
        }
        
        /* Navigation tabs */
        .nav-tabs {
            display: flex;
            gap: 20px;
            margin: 20px 0;
            padding: 10px 0;
            border-bottom: 1px solid #333;
        }
        .nav-tab {
            color: #e5e5e5;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            padding: 5px 0;
            transition: color 0.3s;
        }
        .nav-tab:hover {
            color: #b3b3b3;
        }
        .nav-tab.active {
            color: #ffffff;
            font-weight: 600;
            border-bottom: 2px solid #e50914;
        }
        
        /* Category headers */
        .category-header {
            color: #e5e5e5;
            font-size: 1.5rem;
            font-weight: 600;
            margin: 30px 0 15px 0;
            letter-spacing: -0.5px;
        }
        
        /* Horizontal scroll container */
        .movies-row {
            display: flex;
            overflow-x: auto;
            gap: 10px;
            padding: 10px 0 20px 0;
            scroll-behavior: smooth;
        }
        .movies-row::-webkit-scrollbar {
            display: none;
        }
        
        /* Movie card - Netflix style */
        .movie-card {
            flex: 0 0 auto;
            width: 230px;
            transition: transform 0.3s;
            position: relative;
            cursor: pointer;
        }
        .movie-card:hover {
            transform: scale(1.05);
            z-index: 2;
        }
        
        /* Movie poster */
        .movie-poster {
            width: 100%;
            height: 345px;
            object-fit: cover;
            border-radius: 4px;
            background: linear-gradient(to bottom, #333, #141414);
        }
        
        /* Rating badge */
        .rating-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: #ffd700;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9rem;
            font-weight: 600;
        }
        
        /* Year badge */
        .year-badge {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(229, 9, 20, 0.9);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        /* Movie info */
        .movie-info {
            margin-top: 10px;
        }
        .movie-title {
            color: #ffffff;
            font-size: 1rem;
            font-weight: 500;
            margin: 5px 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .movie-meta {
            color: #a3a3a3;
            font-size: 0.85rem;
            display: flex;
            gap: 10px;
        }
        
        /* Featured hero section */
        .hero-section {
            background: linear-gradient(to bottom, rgba(20,20,20,0.9), rgba(20,20,20,0.95)), url('https://via.placeholder.com/1920x600');
            padding: 80px 0 40px 40px;
            margin: -80px -60px 20px -60px;
            border-bottom: 1px solid #333;
        }
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 10px;
        }
        .hero-meta {
            display: flex;
            gap: 15px;
            color: #a3a3a3;
            font-size: 1.1rem;
            margin-bottom: 20px;
        }
        .hero-rating {
            color: #ffd700;
        }
        .hero-description {
            color: #e5e5e5;
            font-size: 1.2rem;
            max-width: 600px;
            margin-bottom: 30px;
        }
        .watch-btn {
            background-color: #e50914;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 1.2rem;
            font-weight: 600;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
        }
        .watch-btn:hover {
            background-color: #f40612;
        }
        
        /* Filter chips */
        .filter-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
        }
        .filter-chip {
            background: #333;
            color: #e5e5e5;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s;
        }
        .filter-chip:hover {
            background: #4d4d4d;
        }
        .filter-chip.active {
            background: #e50914;
            color: white;
        }
        
        /* Search bar */
        .search-container {
            margin: 20px 0;
        }
        .stTextInput input {
            background-color: #333 !important;
            border: 1px solid #4d4d4d !important;
            color: white !important;
            border-radius: 4px !important;
            padding: 12px !important;
        }
        .stTextInput input:focus {
            border-color: #e50914 !important;
            box-shadow: none !important;
        }
        
        /* Select box */
        .stSelectbox div {
            background-color: #333 !important;
            border-color: #4d4d4d !important;
            color: white !important;
        }
        
        /* Hide default labels */
        .stTextInput label, .stSelectbox label {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS FOR UI ---
def get_featured_movie():
    """Get a random popular movie for hero section"""
    popular_movies = movies.nlargest(10, 'vote_count')
    if not popular_movies.empty:
        return popular_movies.iloc[0]['title']
    return movies.iloc[0]['title']

def render_movie_row(recommendations, title, score_column=None):
    """Render a horizontal row of movie cards"""
    if recommendations.empty:
        return
    
    st.markdown(f'<div class="category-header">{title}</div>', unsafe_allow_html=True)
    
    html = '<div class="movies-row">'
    
    for _, row in recommendations.head(15).iterrows():  # Show up to 15 movies
        poster_url, overview, movie_link = fetch_movie_details(row['title'])
        
        # Get rating
        if score_column and score_column in row:
            rating = f"{row[score_column]:.1f}%"
        else:
            rating = f"{row.get('vote_average', 0)*10:.0f}%"
        
        # Get year (if available)
        year = ""
        if 'release_date' in row and pd.notna(row['release_date']):
            year = str(row['release_date'])[:4]
        
        html += f'''
            <div class="movie-card">
                <img src="{poster_url}" class="movie-poster" alt="{row['title']}">
                <div class="rating-badge">⭐ {rating}</div>
                {f'<div class="year-badge">{year}</div>' if year else ''}
                <div class="movie-info">
                    <div class="movie-title">{row['title']}</div>
                    <div class="movie-meta">
                        <span>{row.get('genres_clean', '').split()[0] if pd.notna(row.get('genres_clean')) else ''}</span>
                        <span>•</span>
                        <span>{year if year else 'N/A'}</span>
                    </div>
                </div>
            </div>
        '''
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# --- MAIN APP LAYOUT ---

# Header with Netflix-style logo and nav
col1, col2, col3 = st.columns([1, 3, 2])
with col1:
    st.markdown('<p class="netflix-logo">CINEMATCH</p>', unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class="nav-tabs">
            <span class="nav-tab active">Home</span>
            <span class="nav-tab">Movies</span>
            <span class="nav-tab">Series</span>
            <span class="nav-tab">My List</span>
        </div>
    """, unsafe_allow_html=True)
with col3:
    # Search bar
    search_query = st.text_input("", placeholder="Search movies, genres, actors...", key="search")

# Filter chips (like in the image)
st.markdown("""
    <div class="filter-chips">
        <span class="filter-chip active">All</span>
        <span class="filter-chip">Action</span>
        <span class="filter-chip">Adventure</span>
        <span class="filter-chip">Comedy</span>
        <span class="filter-chip">Crime</span>
        <span class="filter-chip">Documentary</span>
        <span class="filter-chip">Drama</span>
        <span class="filter-chip">Year</span>
        <span class="filter-chip">A-Z</span>
    </div>
""", unsafe_allow_html=True)

# Model selector (moved to sidebar or hidden)
with st.sidebar:
    st.markdown("## Recommendation Engine")
    selected_display = st.selectbox(
        "Choose Model:",
        ["✨ Hybrid Top Picks", "👥 Community Picks", "🎭 AI Similar (Content-Based)"],
        label_visibility="collapsed"
    )

# Main content
if search_query:
    # Search results logic
    with st.spinner('Finding your perfect matches...'):
        query_vec = tfidf.transform([search_query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_match_idx = sim_scores.argmax()
        best_score = sim_scores[best_match_idx]
        
        if best_score > 0.15:
            selected_movie = movies.iloc[best_match_idx]['title']
            
            # Hero section for selected movie
            poster_url, overview, movie_link = fetch_movie_details(selected_movie)
            st.markdown(f'''
                <div class="hero-section" style="background: linear-gradient(to bottom, rgba(20,20,20,0.9), rgba(20,20,20,0.95)), url('{poster_url}'); background-size: cover; background-position: center;">
                    <div class="hero-title">{selected_movie}</div>
                    <div class="hero-meta">
                        <span class="hero-rating">⭐ {movies[movies['title']==selected_movie]['vote_average'].values[0]*10:.1f}% Match</span>
                        <span>•</span>
                        <span>{movies[movies['title']==selected_movie]['release_date'].values[0][:4] if pd.notna(movies[movies['title']==selected_movie]['release_date'].values[0]) else '2024'}</span>
                        <span>•</span>
                        <span>{movies[movies['title']==selected_movie]['genres_clean'].values[0].split()[0] if pd.notna(movies[movies['title']==selected_movie]['genres_clean'].values[0]) else 'Drama'}</span>
                    </div>
                    <div class="hero-description">{overview}</div>
                    <a href="{movie_link}" target="_blank" class="watch-btn">▶ WATCH NOW</a>
                </div>
            ''', unsafe_allow_html=True)
            
            # Recommendation rows
            if selected_display == "✨ Hybrid Top Picks" or "Show All Rows" in selected_display:
                render_movie_row(get_hybrid_recs(selected_movie), "Trending Now", 'Hybrid_Score')
            
            if selected_display == "👥 Community Picks" or "Show All Rows" in selected_display:
                render_movie_row(get_community_recs(selected_movie), "Popular Netflix Original Premieres", 'CF_Score')
            
            if selected_display == "🎭 AI Similar (Content-Based)" or "Show All Rows" in selected_display:
                render_movie_row(get_content_based_recs(selected_movie), "Recently Added", 'CB_Score')
        
        else:
            # Topic search results
            st.markdown(f'<div class="category-header">Search Results for "{search_query}"</div>', unsafe_allow_html=True)
            topic_results = search_tmdb_topic(search_query)
            if topic_results:
                topic_df = pd.DataFrame(topic_results)
                render_movie_row(topic_df, "Movies Found", 'score')
            else:
                st.warning("No movies found for that topic.")

else:
    # Default home page with featured content
    featured = get_featured_movie()
    poster_url, overview, movie_link = fetch_movie_details(featured)
    
    st.markdown(f'''
        <div class="hero-section" style="background: linear-gradient(to bottom, rgba(20,20,20,0.9), rgba(20,20,20,0.95)), url('{poster_url}'); background-size: cover; background-position: center;">
            <div class="hero-title">{featured}</div>
            <div class="hero-meta">
                <span class="hero-rating">⭐ {movies[movies['title']==featured]['vote_average'].values[0]*10:.1f}% Match</span>
                <span>•</span>
                <span>{movies[movies['title']==featured]['release_date'].values[0][:4] if pd.notna(movies[movies['title']==featured]['release_date'].values[0]) else '2024'}</span>
                <span>•</span>
                <span>2 Seasons</span>
            </div>
            <div class="hero-description">{overview[:200]}...</div>
            <a href="{movie_link}" target="_blank" class="watch-btn">▶ WATCH NOW</a>
        </div>
    ''', unsafe_allow_html=True)
    
    # Show different categories
    render_movie_row(movies.sample(15), "Trending Now", 'vote_average')
    render_movie_row(movies.sample(15), "Popular Netflix Original Premieres", 'vote_average')
    render_movie_row(movies.sample(15), "Recently Added", 'vote_average')

# Footer with info
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>© 2024 CineMatch Pro. All rights reserved. | Privacy | Terms | Contact</p>
    </div>
""", unsafe_allow_html=True)
