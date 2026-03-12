import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random

# --- Import logic from movie_logic.py ---
from movie_logic import (
    movies, cosine_sim, tfidf, tfidf_matrix,
    fetch_movie_details, search_tmdb_topic,
    get_content_based_recs, get_community_recs, get_hybrid_recs
)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Zmovo - Stream Smarter", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

# --- ENHANCED CSS with Animations & Premium Effects ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        color: #ffffff;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu, footer, header {
        display: none !important;
    }
    
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Animated Gradient Search Bar */
    div[data-testid="stTextInput"] {
        position: fixed;
        top: 30px;
        right: 4%;
        z-index: 9999;
        width: 400px;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    div[data-testid="stTextInput"] input {
        background: rgba(20, 20, 20, 0.8) !important;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 40px !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 15px 25px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    div[data-testid="stTextInput"] input:hover {
        border-color: rgba(229, 9, 20, 0.5) !important;
        background: rgba(30, 30, 30, 0.9) !important;
    }
    
    div[data-testid="stTextInput"] input:focus {
        border-color: #e50914 !important;
        background: rgba(20, 20, 20, 0.95) !important;
        box-shadow: 0 4px 30px rgba(229, 9, 20, 0.3) !important;
    }
    
    /* Hero Banner with Parallax Effect */
    .hero-banner {
        height: 95vh;
        position: relative;
        display: flex;
        align-items: center;
        padding: 0 6%;
        background-size: cover !important;
        background-position: center !important;
        animation: fadeIn 1s ease-in;
        transform-origin: center;
        transition: transform 0.1s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .hero-vignette {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(77deg, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0.5) 50%, rgba(0,0,0,0.2) 100%),
                    radial-gradient(circle at 20% 50%, rgba(0,0,0,0) 0%, rgba(0,0,0,0.6) 100%);
        pointer-events: none;
    }
    
    .hero-bottom-fade {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 200px;
        background: linear-gradient(to top, #0a0a0a 0%, transparent 100%);
        pointer-events: none;
    }
    
    .hero-content {
        position: relative;
        z-index: 10;
        max-width: 700px;
        animation: slideUp 0.8s ease-out 0.2s both;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .hero-title {
        font-size: 5rem;
        font-weight: 900;
        margin-bottom: 20px;
        background: linear-gradient(to right, #ffffff, #e5e5e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(255,255,255,0.3);
        line-height: 1.1;
    }
    
    .hero-meta {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.8);
        margin-bottom: 30px;
        line-height: 1.6;
        max-width: 600px;
        display: -webkit-box;
        -webkit-line-clamp: 4;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    /* Premium Buttons */
    .btn-row {
        display: flex;
        gap: 15px;
        margin-top: 20px;
    }
    
    .btn-play, .btn-info {
        padding: 14px 40px;
        border-radius: 40px;
        font-weight: 700;
        font-size: 1.2rem;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        border: none;
    }
    
    .btn-play {
        background: linear-gradient(45deg, #e50914, #ff5f6d);
        color: white;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.3);
    }
    
    .btn-play:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(229, 9, 20, 0.5);
    }
    
    .btn-info {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .btn-info:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: scale(1.05);
    }
    
    /* Section Styles */
    .section-container {
        padding: 0 6%;
        margin-top: -60px;
        position: relative;
        z-index: 20;
    }
    
    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 20px;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(to right, #ffffff, #b3b3b3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    
    .section-link {
        color: rgba(255,255,255,0.6);
        text-decoration: none;
        font-size: 0.9rem;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    
    .section-link:hover {
        color: #e50914;
    }
    
    /* Enhanced Scroll Container */
    .scroll-container {
        display: flex;
        flex-wrap: nowrap;
        overflow-x: auto;
        gap: 15px;
        padding: 20px 0 40px 0;
        scroll-behavior: smooth;
        -ms-overflow-style: none;
        scrollbar-width: none;
    }
    
    .scroll-container::-webkit-scrollbar {
        display: none;
    }
    
    /* Premium Movie Cards */
    .movie-card {
        flex: 0 0 300px;
        aspect-ratio: 16/9;
        border-radius: 8px;
        overflow: hidden;
        position: relative;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 2px solid transparent;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    }
    
    .movie-card:hover {
        transform: scale(1.1) translateY(-10px);
        border-color: #e50914;
        box-shadow: 0 20px 30px rgba(229, 9, 20, 0.3);
        z-index: 30;
    }
    
    .movie-card img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.4s ease;
    }
    
    .movie-card:hover img {
        transform: scale(1.1);
    }
    
    /* Card Overlay */
    .card-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(to top, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0.2) 60%);
        opacity: 0;
        transition: opacity 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 20px;
    }
    
    .movie-card:hover .card-overlay {
        opacity: 1;
    }
    
    .card-match {
        color: #46d369;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 5px;
        animation: slideUp 0.3s ease;
    }
    
    .card-title {
        color: white;
        font-weight: 700;
        font-size: 1rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        animation: slideUp 0.3s ease 0.05s both;
    }
    
    /* Top 10 Cards */
    .top10-wrapper {
        flex: 0 0 350px;
        display: flex;
        position: relative;
        cursor: pointer;
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .top10-wrapper:hover {
        transform: scale(1.05) translateY(-10px);
        z-index: 30;
    }
    
    .top10-number {
        font-size: 16rem;
        font-weight: 900;
        color: transparent;
        -webkit-text-stroke: 4px rgba(229, 9, 20, 0.5);
        position: absolute;
        left: -40px;
        bottom: -70px;
        z-index: 1;
        line-height: 1;
        transition: all 0.3s ease;
    }
    
    .top10-wrapper:hover .top10-number {
        -webkit-text-stroke-color: #e50914;
        transform: scale(1.1);
    }
    
    .top10-card {
        flex: 1;
        aspect-ratio: 2/3;
        border-radius: 8px;
        overflow: hidden;
        position: relative;
        z-index: 2;
        margin-left: 60px;
        box-shadow: 0 15px 30px rgba(0,0,0,0.5);
    }
    
    .top10-card img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    /* Badges */
    .badge-top10 {
        position: absolute;
        top: 10px;
        right: 10px;
        background: linear-gradient(45deg, #e50914, #ff5f6d);
        color: white;
        font-size: 0.7rem;
        font-weight: 900;
        padding: 4px 8px;
        border-radius: 4px;
        z-index: 10;
        box-shadow: 0 2px 10px rgba(229, 9, 20, 0.5);
    }
    
    .badge-recent {
        position: absolute;
        bottom: 10px;
        left: 10px;
        background: rgba(0,0,0,0.8);
        backdrop-filter: blur(5px);
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.2);
        z-index: 10;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }
    
    .loading-spinner::after {
        content: '';
        width: 50px;
        height: 50px;
        border: 5px solid rgba(229, 9, 20, 0.2);
        border-top-color: #e50914;
        border-radius: 50%;
        animation: spinner 1s linear infinite;
    }
    
    @keyframes spinner {
        to { transform: rotate(360deg); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 3rem;
        }
        
        .hero-meta {
            font-size: 1rem;
        }
        
        .section-title {
            font-size: 1.4rem;
        }
        
        .movie-card {
            flex: 0 0 250px;
        }
        
        div[data-testid="stTextInput"] {
            width: 300px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- RENDER MOVIE CARDS FUNCTION ---
def render_movie_cards(recommendations, score_column, is_top10_row=False, show_recent=False, show_top10_badge=False):
    html_content = '<div class="scroll-container">'
    
    for i, (_, row) in enumerate(recommendations.iterrows()):
        poster_url, overview, movie_link = fetch_movie_details(row['title'])
        score = row.get(score_column, 85)
        
        recent_badge = '<div class="badge-recent">✨ New</div>' if show_recent else ''
        top10_badge = '<div class="badge-top10">TOP 10</div>' if show_top10_badge else ''

        if is_top10_row:
            html_content += f"""
            <div class="top10-wrapper" onclick="window.open('{movie_link}', '_blank')">
                <div class="top10-number">{i+1}</div>
                <div class="top10-card">
                    <img src="{poster_url}" alt="{row['title']}">
                    {recent_badge}
                </div>
            </div>"""
        else:
            html_content += f"""
            <div class="movie-card" onclick="window.open('{movie_link}', '_blank')">
                <img src="{poster_url}" alt="{row['title']}">
                {top10_badge}
                {recent_badge}
                <div class="card-overlay">
                    <div class="card-match">{score:.0f}% Match</div>
                    <div class="card-title">{row['title']}</div>
                </div>
            </div>"""
    
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

# --- SEARCH INPUT ---
search_query = st.text_input("", placeholder="🔍 Search for movies, shows, genres...", label_visibility="collapsed")

# --- MAIN LOGIC ---
if search_query:
    with st.spinner(''):
        # Show loading animation
        st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
        
        query_vec = tfidf.transform([search_query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_match_idx = sim_scores.argmax()
        best_score = sim_scores[best_match_idx]
        
        if best_score > 0:
            selected_movie = movies.iloc[best_match_idx]['title']
            poster_url, overview, _ = fetch_movie_details(selected_movie)
            
            # Enhanced Hero Banner
            hero_html = f"""
            <div class="hero-banner" style="background-image: url('{poster_url}');">
                <div class="hero-vignette"></div>
                <div class="hero-bottom-fade"></div>
                <div class="hero-content">
                    <h1 class="hero-title">{selected_movie}</h1>
                    <p class="hero-meta">{overview}</p>
                    <div class="btn-row">
                        <a href="#" class="btn-play">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M8 5V19L19 12L8 5Z" fill="currentColor"/>
                            </svg>
                            Play Now
                        </a>
                        <a href="#" class="btn-info">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                                <circle cx="12" cy="16" r="1.5" fill="currentColor"/>
                                <path d="M12 8V13" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                            </svg>
                            More Info
                        </a>
                    </div>
                </div>
            </div>
            """
            st.markdown(hero_html, unsafe_allow_html=True)
            
            # Content Sections
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            
            # Similar Movies
            st.markdown("""
                <div class="section-header">
                    <div class="section-title">More Like This</div>
                    <a href="#" class="section-link">Browse All →</a>
                </div>
            """, unsafe_allow_html=True)
            render_movie_cards(get_hybrid_recs(selected_movie), 'Hybrid_Score')
            
            # Trending Now
            st.markdown("""
                <div class="section-header">
                    <div class="section-title">🔥 Trending Now</div>
                    <a href="#" class="section-link">View All →</a>
                </div>
            """, unsafe_allow_html=True)
            render_movie_cards(get_community_recs(selected_movie), 'CF_Score', show_recent=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            # Search Results
            st.markdown('<div class="section-container" style="margin-top: 120px;">', unsafe_allow_html=True)
            st.markdown("""
                <div class="section-header">
                    <div class="section-title">🔍 Search Results</div>
                </div>
            """, unsafe_allow_html=True)
            
            topic_results = search_tmdb_topic(search_query)
            if topic_results:
                render_movie_cards(pd.DataFrame(topic_results), 'score')
            else:
                st.markdown("""
                    <div style="text-align: center; padding: 100px 0;">
                        <h2 style="color: #666;">No results found for "{}"</h2>
                        <p style="color: #999;">Try searching for something else</p>
                    </div>
                """.format(search_query), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

else:
    # Default Homepage
    # Hero Banner
    hero_image = "https://images.unsplash.com/photo-1626814026160-2237a95fc5a0?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"
    st.markdown(f"""
        <div class="hero-banner" style="background-image: url('{hero_image}');">
            <div class="hero-vignette"></div>
            <div class="hero-bottom-fade"></div>
            <div class="hero-content">
                <h1 class="hero-title">WELCOME TO ZMOVO</h1>
                <p class="hero-meta">Experience the ultimate streaming destination. Watch blockbuster movies, exclusive originals, and trending shows in stunning 4K HDR.</p>
                <div class="btn-row">
                    <a href="#" class="btn-play">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M8 5V19L19 12L8 5Z" fill="currentColor"/>
                        </svg>
                        Start Watching
                    </a>
                    <a href="#" class="btn-info">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                            <circle cx="12" cy="16" r="1.5" fill="currentColor"/>
                            <path d="M12 8V13" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                        </svg>
                        Learn More
                    </a>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Content Sections
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    
    # Top 10 Section
    st.markdown("""
        <div class="section-header">
            <div class="section-title">🇲🇾 Top 10 in Malaysia Today</div>
            <a href="#" class="section-link">See All →</a>
        </div>
    """, unsafe_allow_html=True)
    render_movie_cards(movies.sort_values('vote_count', ascending=False).head(10), 'vote_average', is_top10_row=True, show_recent=True)
    
    # New Releases
    st.markdown("""
        <div class="section-header">
            <div class="section-title">🎉 New on Zmovo</div>
            <a href="#" class="section-link">Explore New Releases →</a>
        </div>
    """, unsafe_allow_html=True)
    render_movie_cards(movies.head(10), 'vote_average', show_top10_badge=True)
    
    # Popular Movies
st.markdown("""
    <div class="section-header">
        <div class="section-title">⭐ Popular Movies</div>
        <a href="#" class="section-link">View All →</a>
    </div>
""", unsafe_allow_html=True)

# Sort by vote_count (most voted) or vote_average (highest rated)
render_movie_cards(
    movies.sort_values('vote_count', ascending=False).head(10), 
    'vote_average', 
    show_recent=False
)
    
    st.markdown('</div>', unsafe_allow_html=True)
