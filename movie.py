import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random

# --- Import logic from movie_logic.py ---
# Note: Ensure this module exists in your actual directory
from movie_logic import (
    movies, cosine_sim, tfidf, tfidf_matrix,
    fetch_movie_details, search_tmdb_topic,
    get_content_based_recs, get_community_recs, get_hybrid_recs
)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Zmovo - Stream Smarter", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

# --- ENHANCED CSS TO MATCH NETFLIX UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Helvetica+Neue:wght@300;400;500;700;900&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background-color: #141414;
        color: #ffffff;
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu, footer, header { display: none !important; }
    .block-container { padding: 0 !important; max-width: 100% !important; overflow-x: hidden; }
    
    /* Fake Netflix Navbar */
    .netflix-navbar {
        position: fixed;
        top: 0;
        width: 100%;
        height: 68px;
        background: linear-gradient(to bottom, rgba(0,0,0,0.7) 10%, rgba(0,0,0,0));
        z-index: 999;
        display: flex;
        align-items: center;
        padding: 0 4%;
        gap: 30px;
    }
    .netflix-logo { color: #E50914; font-size: 1.8rem; font-weight: 900; letter-spacing: 1px; }
    .nav-links a { color: #e5e5e5; text-decoration: none; font-size: 0.85rem; font-weight: 500; transition: color 0.4s; margin-right: 18px; }
    .nav-links a:hover { color: #b3b3b3; }
    .nav-links a.active { font-weight: 700; color: white; }
    
    /* Sub Header (Movies Dropdown) */
    .sub-header {
        position: absolute;
        top: 80px;
        left: 4%;
        z-index: 100;
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .sub-header h2 { font-size: 2.2rem; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.45); }
    .genre-dropdown { background: rgba(0,0,0,0.8); border: 1px solid rgba(255,255,255,0.2); color: white; padding: 4px 10px; font-weight: bold; cursor: pointer; }
    
    /* Search Bar Tweaks */
    div[data-testid="stTextInput"] {
        position: fixed; top: 15px; right: 4%; z-index: 1000; width: 250px;
    }
    div[data-testid="stTextInput"] input {
        background: rgba(0, 0, 0, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important; padding: 8px 15px !important;
    }
    
    /* Hero Banner */
    .hero-banner {
        height: 85vh;
        position: relative;
        display: flex;
        align-items: center;
        padding: 0 4%;
        background-size: cover !important;
        background-position: center 10% !important;
    }
    
    .hero-vignette {
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(77deg, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0.4) 40%, transparent 85%);
    }
    
    .hero-bottom-fade {
        position: absolute; bottom: 0; left: 0; right: 0; height: 150px;
        background: linear-gradient(to top, #141414 0%, transparent 100%);
    }
    
    .hero-content {
        position: relative; z-index: 10; max-width: 600px; padding-top: 50px;
    }
    
    .hero-title {
        font-size: 5rem; font-weight: 900; margin-bottom: 10px;
        text-transform: uppercase; font-family: Impact, sans-serif;
        letter-spacing: 2px; text-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    
    .hero-top10-badge {
        display: flex; align-items: center; gap: 10px; margin-bottom: 15px; font-weight: bold; font-size: 1.2rem; text-shadow: 1px 1px 2px black;
    }
    .hero-top10-icon {
        background-color: #E50914; color: white; padding: 2px 6px; font-size: 0.8rem; border-radius: 2px;
    }
    
    .hero-meta { font-size: 1.2rem; color: #fff; margin-bottom: 25px; line-height: 1.4; text-shadow: 1px 1px 2px black; }
    
    /* Buttons matching image exactly */
    .btn-row { display: flex; gap: 10px; }
    
    .btn-play, .btn-info {
        padding: 10px 24px 10px 20px;
        border-radius: 4px; font-weight: bold; font-size: 1.2rem;
        text-decoration: none; display: inline-flex; align-items: center; gap: 10px;
        transition: all 0.2s; border: none; cursor: pointer;
    }
    
    .btn-play { background: white; color: black; }
    .btn-play:hover { background: rgba(255, 255, 255, 0.7); }
    
    .btn-info { background: rgba(109, 109, 110, 0.7); color: white; }
    .btn-info:hover { background: rgba(109, 109, 110, 0.4); }
    
    /* Section Styles */
    .section-container { padding: 0 4%; position: relative; z-index: 20; margin-top: -30px; }
    .section-title { font-size: 1.2vw; font-weight: 500; color: #e5e5e5; margin-bottom: 10px; margin-top: 3vw; }
    
    /* Scroll Containers */
    .scroll-container {
        display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 8px;
        padding-bottom: 20px; scroll-behavior: smooth; scrollbar-width: none;
    }
    .scroll-container::-webkit-scrollbar { display: none; }
    
    /* Standard Landscape Card */
    .movie-card {
        flex: 0 0 240px; aspect-ratio: 16/9; border-radius: 4px; overflow: hidden;
        position: relative; cursor: pointer; transition: transform 0.3s ease;
        background-color: #222;
    }
    .movie-card:hover { transform: scale(1.05); z-index: 5; }
    .movie-card img { width: 100%; height: 100%; object-fit: cover; }
    
    /* Top 10 Giant Numbers Card Styles */
    .top10-wrapper {
        flex: 0 0 220px; height: 210px; display: flex; align-items: flex-end;
        position: relative; cursor: pointer; overflow: visible;
    }
    .top10-number {
        font-size: 220px; font-weight: 900;
        color: #000; -webkit-text-stroke: 4px #595959;
        line-height: 0.8; letter-spacing: -10px;
        position: absolute; left: 0; bottom: -15px; z-index: 1;
    }
    .top10-card {
        width: 140px; aspect-ratio: 2/3; border-radius: 4px;
        overflow: hidden; position: absolute; right: 0; bottom: 0; z-index: 2;
        box-shadow: -5px 0 15px rgba(0,0,0,0.5); transition: transform 0.3s ease;
    }
    .top10-wrapper:hover .top10-card { transform: scale(1.05); }
    .top10-card img { width: 100%; height: 100%; object-fit: cover; }

    /* Netflix Top Red Badge */
    .badge-top-left {
        position: absolute; top: 0; left: 5px; width: 25px; z-index: 10;
    }
    </style>
""", unsafe_allow_html=True)

# --- FAKE NAVBAR HTML ---
st.markdown("""
    <div class="netflix-navbar">
        <div class="netflix-logo">ZMOVO</div>
        <div class="nav-links">
            <a href="#">Home</a>
            <a href="#">Shows</a>
            <a href="#" class="active">Movies</a>
            <a href="#">Games</a>
            <a href="#">New & Popular</a>
            <a href="#">My List</a>
            <a href="#">Browse by Languages</a>
        </div>
    </div>
    <div class="sub-header">
        <h2>Movies</h2>
        <select class="genre-dropdown"><option>Genres ▾</option></select>
    </div>
""", unsafe_allow_html=True)


# --- RENDER MOVIE CARDS FUNCTION ---
def render_movie_cards(recommendations, is_top10_row=False):
    html_content = '<div class="scroll-container">'
    
    for i, (_, row) in enumerate(recommendations.iterrows()):
        try:
            poster_url, overview, movie_link = fetch_movie_details(row['title'])
            
            # Simulated Netflix "N" ribbon
            netflix_badge = '<img src="https://upload.wikimedia.org/wikipedia/commons/0/0c/Netflix_2015_N_logo.svg" class="badge-top-left">'
            
            if is_top10_row:
                # Layout for the Giant Number Row
                html_content += f"""
                <div class="top10-wrapper" onclick="window.open('{movie_link}', '_blank')">
                    <div class="top10-number">{i+1}</div>
                    <div class="top10-card">
                        {netflix_badge}
                        <img src="{poster_url}" alt="{row['title']}">
                    </div>
                </div>"""
            else:
                # Layout for standard Landscape Row
                # For a landscape image, tmdb usually provides backdrop_paths. If fetch_movie_details only fetches posters,
                # you might want to modify your movie_logic to fetch backdrops for these rows. 
                # Assuming poster_url serves as the image here:
                html_content += f"""
                <div class="movie-card" onclick="window.open('{movie_link}', '_blank')">
                    {netflix_badge}
                    <img src="{poster_url}" alt="{row['title']}">
                </div>"""
        except Exception:
            continue
            
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)


# --- SEARCH INPUT ---
search_query = st.text_input("", placeholder="🔍 Titles, people, genres", label_visibility="collapsed")


# --- MAIN UI LOGIC ---
if search_query:
    with st.spinner(''):
        query_vec = tfidf.transform([search_query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_match_idx = sim_scores.argmax()
        
        if sim_scores[best_match_idx] > 0:
            selected_movie = movies.iloc[best_match_idx]['title']
            poster_url, overview, _ = fetch_movie_details(selected_movie)
            
            # Simple hero for search result
            st.markdown(f"""
            <div class="hero-banner" style="background-image: url('{poster_url}'); height: 60vh;">
                <div class="hero-vignette"></div><div class="hero-bottom-fade"></div>
                <div class="hero-content">
                    <h1 class="hero-title">{selected_movie}</h1>
                    <p class="hero-meta">{overview[:150]}...</p>
                </div>
            </div>
            <div class="section-container">
                <div class="section-title">Similar to {selected_movie}</div>
            """, unsafe_allow_html=True)
            render_movie_cards(get_hybrid_recs(selected_movie))
            st.markdown("</div>", unsafe_allow_html=True)

else:
    # --- DEFAULT HOMEPAGE ---
    
    # Hero Banner (Matching the BANDUAN image)
    hero_image = "https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80" # Placeholder for hero backdrop
    st.markdown(f"""
        <div class="hero-banner" style="background-image: url('{hero_image}');">
            <div class="hero-vignette"></div>
            <div class="hero-bottom-fade"></div>
            <div class="hero-content">
                <h1 class="hero-title">ZMOVO</h1>
                <div class="hero-top10-badge">
                    <span class="hero-top10-icon">TOP<br>10</span>
                    #1 in Movies Today
                </div>
                <p class="hero-meta">Desperate to meet his young daughter, a newly freed ex-con must survive a night of violence after he's forced to protect police from a ruthless gang.</p>
                <div class="btn-row">
                    <button class="btn-play">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M8 5v14l11-7z"/></svg>
                        Play
                    </button>
                    <button class="btn-info">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
                        More Info
                    </button>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Content Rows
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    
    # Row 1: Top 10 Giant Numbers
    st.markdown('<div class="section-title">Top 10 Movies in Malaysia Today</div>', unsafe_allow_html=True)
    render_movie_cards(movies.sort_values('vote_count', ascending=False).head(10), is_top10_row=True)
    
    # Row 2: Standard Landscape Posters
    st.markdown('<div class="section-title">New on Zmovo</div>', unsafe_allow_html=True)
    render_movie_cards(movies.head(10), is_top10_row=False)
    
    # Row 3: We Think You'll Love
