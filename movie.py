import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --- MAGIC STEP: Import our functions from our new logic file ---
from movie_logic import (
    movies, cosine_sim, tfidf, tfidf_matrix,
    fetch_movie_details, search_tmdb_topic,
    get_content_based_recs, get_community_recs, get_hybrid_recs
)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CineMatch Pro", page_icon="🍿", layout="wide")

# --- CSS STYLING (Netflix Dark Theme) ---
st.markdown("""
    <style>
    /* Force Dark Background */
    .stApp { background-color: #141414 !important; color: #ffffff !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    
    /* Hide Streamlit default headers and footers */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    /* Top Navigation Bar */
    .navbar { display: flex; align-items: center; padding: 20px 4%; background: linear-gradient(to bottom, rgba(0,0,0,0.8) 0%, rgba(20,20,20,0) 100%); margin-bottom: -60px; position: relative; z-index: 10; }
    .logo { color: #E50914; font-size: 28px; font-weight: 900; letter-spacing: 1px; margin-right: 40px; }
    .nav-links { display: flex; gap: 20px; color: #e5e5e5; font-size: 14px; font-weight: 500; }
    .nav-links span { cursor: pointer; transition: color 0.3s; }
    .nav-links span:hover { color: #ffffff; font-weight: bold; }
    
    /* Hero Banner Section */
    .hero-container { position: relative; width: 100%; height: 65vh; display: flex; align-items: center; margin-bottom: 20px; overflow: hidden; border-radius: 10px;}
    .hero-bg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; opacity: 0.4; filter: blur(3px); mask-image: linear-gradient(to top, rgba(20,20,20,1), rgba(20,20,20,0)); -webkit-mask-image: linear-gradient(to top, transparent, black); }
    .hero-content { position: relative; z-index: 2; padding: 0 4%; max-width: 60%; }
    .hero-title { font-size: 4rem; font-weight: 900; margin-bottom: 10px; line-height: 1.1; text-transform: uppercase; text-shadow: 2px 2px 4px rgba(0,0,0,0.8); }
    .hero-badge { background-color: #E50914; color: white; padding: 4px 8px; border-radius: 3px; font-weight: bold; font-size: 0.9rem; margin-bottom: 15px; display: inline-block; }
    .hero-desc { font-size: 1.2rem; color: #ffffff; text-shadow: 1px 1px 3px rgba(0,0,0,0.8); margin-bottom: 25px; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
    .hero-buttons button { padding: 10px 24px; font-size: 1.2rem; font-weight: bold; border-radius: 4px; border: none; cursor: pointer; margin-right: 15px; }
    .btn-play { background-color: #ffffff; color: #000000; }
    .btn-play:hover { background-color: #e6e6e6; }
    .btn-info { background-color: rgba(109, 109, 110, 0.7); color: white; }
    .btn-info:hover { background-color: rgba(109, 109, 110, 0.4); }

    /* Category Titles */
    .category-header { font-size: 1.4rem; color: #e5e5e5; font-weight: 700; margin-top: 30px; margin-bottom: 10px; padding-left: 4%; }

    /* Horizontal Scrolling Container */
    .scroll-container { display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 15px; padding: 10px 4% 40px 4%; scroll-behavior: smooth; }
    .scroll-container::-webkit-scrollbar { height: 0px; background: transparent; } /* Hide scrollbar for cleaner look */
    
    /* Standard Movie Cards */
    .movie-card { flex: 0 0 180px; position: relative; transition: transform 0.3s ease, z-index 0.3s ease; cursor: pointer; }
    .movie-card:hover { transform: scale(1.05); z-index: 10; }
    .movie-card img { width: 100%; border-radius: 4px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); }
    
    /* Top 10 Specific Cards */
    .top10-card { flex: 0 0 220px; display: flex; align-items: center; position: relative; padding-left: 30px; }
    .top10-number { font-size: 180px; font-weight: 900; color: #000; -webkit-text-stroke: 4px #555; position: absolute; left: -20px; bottom: -35px; z-index: 1; letter-spacing: -15px; }
    .top10-card img { width: 130px; border-radius: 4px; z-index: 2; margin-left: 40px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); transition: transform 0.3s; }
    .top10-card:hover img { transform: scale(1.05); }

    /* Forcing black text on inputs for readability */
    .stTextInput input { color: white !important; background-color: rgba(0,0,0,0.5) !important; border: 1px solid #555 !important; }
    </style>
""", unsafe_allow_html=True)

# --- NAVIGATION BAR (Visual only) ---
st.markdown("""
    <div class="navbar">
        <div class="logo">CineMatch</div>
        <div class="nav-links">
            <span>Home</span>
            <span>Shows</span>
            <span>Movies</span>
            <span>New & Popular</span>
            <span>My List</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- HELPER FUNCTION TO RENDER UI CARDS ---
def render_movie_cards(recommendations, score_column, is_top_10=False):
    html_content = '<div class="scroll-container">'
    
    for i, (_, row) in enumerate(recommendations.iterrows()):
        poster_url, overview, movie_link = fetch_movie_details(row['title'])
        score = row.get(score_column, 85) 
        
        # If rendering the Top 10 list, add the giant background numbers
        if is_top_10:
            rank = i + 1
            html_content += f"""
            <div class="top10-card">
                <div class="top10-number">{rank}</div>
                <a href="{movie_link}" target="_blank">
                    <img src="{poster_url}" alt="{row['title']}">
                </a>
            </div>
            """
        # Standard Netflix-style cards (Image only, details on click/hover visually implied)
        else:
            html_content += f"""
            <div class="movie-card" title="{row['title']} - {score:.0f}% Match">
                <a href="{movie_link}" target="_blank">
                    <img src="{poster_url}" alt="{row['title']}">
                </a>
            </div>
            """
            
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

# --- SEARCH BAR ---
# Placed slightly below the navbar using columns to right-align it
col1, col2, col3 = st.columns([6, 3, 1])
with col2:
    search_query = st.text_input("", placeholder="🔍 Titles, people, genres...", label_visibility="collapsed")

# --- RESULTS SECTION ---
if search_query:
    with st.spinner('Curating...'):
        # 1. UNIVERSAL SEARCH (NLP)
        query_vec = tfidf.transform([search_query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        best_match_idx = sim_scores.argmax()
        best_score = sim_scores[best_match_idx]
        
        # 2. DECISION LOGIC: Known Movie or Topic?
        if best_score > 0:
            selected_movie = movies.iloc[best_match_idx]['title']
            
            # Fetch details for the HERO BANNER
            hero_poster, hero_overview, hero_link = fetch_movie_details(selected_movie)
            
            # --- RENDER HERO BANNER ---
            st.markdown(f"""
                <div class="hero-container">
                    <img src="{hero_poster}" class="hero-bg">
                    <div class="hero-content">
                        <div class="hero-title">{selected_movie}</div>
                        <div class="hero-badge">📺 #1 in Movies Today</div>
                        <div class="hero-desc">{hero_overview}</div>
                        <div class="hero-buttons">
                            <a href="{hero_link}" target="_blank"><button class="btn-play">▶ Play</button></a>
                            <button class="btn-info">ⓘ More Info</button>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # --- RENDER ROWS ---
            # Row 1: Top 10 in Area (Community Picks)
            st.markdown('<div class="category-header">Top 10 Movies in Malaysia Today</div>', unsafe_allow_html=True)
            render_movie_cards(get_community_recs(selected_movie).head(10), 'CF_Score', is_top_10=True)

            # Row 2: AI Similar (Content-Based) -> Styled as "New on Netflix" or "More Like This"
            st.markdown(f'<div class="category-header">Because you searched for {selected_movie}</div>', unsafe_allow_html=True)
            render_movie_cards(get_content_based_recs(selected_movie), 'CB_Score')

            # Row 3: We Think You'll Love These (Hybrid Picks)
            st.markdown('<div class="category-header">We Think You\'ll Love These</div>', unsafe_allow_html=True)
            render_movie_cards(get_hybrid_recs(selected_movie), 'Hybrid_Score')

        # 3. GLOBAL FALLBACK: Search TMDB topic if no local match
        else:
            st.warning(f"Searching global TMDB library for topic: '{search_query}'")
            topic_results = search_tmdb_topic(search_query)
            
            if topic_results:
                st.markdown('<div class="category-header">Global Search Results</div>', unsafe_allow_html=True)
                topic_df = pd.DataFrame(topic_results)
                render_movie_cards(topic_df, 'score')
            else:
                st.error("No movies found for that search.")

# Default Dashboard View (when no search is entered)
else:
    # Just a placeholder aesthetic banner to make it look like the image initially
    st.markdown("""
        <div class="hero-container" style="background: linear-gradient(45deg, #111, #333);">
            <div class="hero-content">
                <div class="hero-title" style="color:#555;">SEARCH TO BEGIN</div>
                <div class="hero-desc">Type a movie title or description in the top right to start your personalized movie dashboard.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
