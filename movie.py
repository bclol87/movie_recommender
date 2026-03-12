import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- Import logic from movie_logic.py ---
from movie_logic import (
    movies, cosine_sim, tfidf, tfidf_matrix,
    fetch_movie_details, search_tmdb_topic,
    get_content_based_recs, get_community_recs, get_hybrid_recs
)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Zmovo - Stream Smarter", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

# --- SIMPLIFIED CSS THAT WORKS ---
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: #0a0a0a;
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
    
    /* Search Bar */
    div[data-testid="stTextInput"] {
        position: fixed;
        top: 30px;
        right: 4%;
        z-index: 9999;
        width: 400px;
    }
    
    div[data-testid="stTextInput"] input {
        background: rgba(20, 20, 20, 0.9) !important;
        border: 2px solid #e50914 !important;
        border-radius: 40px !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 15px 25px !important;
    }
    
    /* Hero Banner */
    .hero-banner {
        height: 85vh;
        position: relative;
        display: flex;
        align-items: center;
        padding: 0 6%;
        background-size: cover !important;
        background-position: center !important;
    }
    
    .hero-vignette {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(77deg, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0.5) 50%, rgba(0,0,0,0.2) 100%);
        pointer-events: none;
    }
    
    .hero-bottom-fade {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 150px;
        background: linear-gradient(to top, #0a0a0a 0%, transparent 100%);
        pointer-events: none;
    }
    
    .hero-content {
        position: relative;
        z-index: 10;
        max-width: 700px;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 900;
        margin-bottom: 20px;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .hero-meta {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.8);
        margin-bottom: 30px;
        line-height: 1.6;
    }
    
    /* Buttons */
    .btn-row {
        display: flex;
        gap: 15px;
    }
    
    .btn-play, .btn-info {
        padding: 12px 30px;
        border-radius: 40px;
        font-weight: 600;
        font-size: 1.1rem;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
        border: none;
    }
    
    .btn-play {
        background: #e50914;
        color: white;
    }
    
    .btn-info {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Section Styles */
    .section-container {
        padding: 0 6%;
        margin-top: -40px;
        position: relative;
        z-index: 20;
    }
    
    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
    }
    
    /* Simple Scroll Container - GUARANTEED TO WORK */
    .scroll-container {
        display: flex;
        overflow-x: auto;
        gap: 15px;
        padding: 10px 0 30px 0;
        margin: 0;
        min-height: 180px;
    }
    
    .scroll-container::-webkit-scrollbar {
        display: none;
    }
    
    /* Simple Movie Cards */
    .movie-card {
        flex: 0 0 280px;
        width: 280px;
        height: 158px;
        border-radius: 4px;
        overflow: hidden;
        position: relative;
        cursor: pointer;
        background: #1a1a1a;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .movie-card img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .card-match {
        position: absolute;
        bottom: 10px;
        left: 10px;
        color: #46d369;
        font-weight: 700;
        font-size: 0.9rem;
        background: rgba(0,0,0,0.7);
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    .badge-top10 {
        position: absolute;
        top: 10px;
        right: 10px;
        background: #e50914;
        color: white;
        font-size: 0.7rem;
        font-weight: 900;
        padding: 3px 6px;
        border-radius: 2px;
        z-index: 5;
    }
    
    .badge-recent {
        position: absolute;
        bottom: 10px;
        left: 10px;
        background: rgba(0,0,0,0.7);
        color: white;
        font-size: 0.7rem;
        padding: 3px 8px;
        border-radius: 12px;
        z-index: 5;
    }
    
    /* Top 10 Cards */
    .top10-wrapper {
        flex: 0 0 300px;
        width: 300px;
        display: flex;
        position: relative;
        cursor: pointer;
        height: 200px;
    }
    
    .top10-number {
        font-size: 12rem;
        font-weight: 900;
        color: transparent;
        -webkit-text-stroke: 3px #e50914;
        position: absolute;
        left: -30px;
        bottom: -40px;
        z-index: 1;
        line-height: 1;
    }
    
    .top10-card {
        flex: 1;
        height: 100%;
        border-radius: 4px;
        overflow: hidden;
        position: relative;
        z-index: 2;
        margin-left: 50px;
        background: #1a1a1a;
    }
    
    .top10-card img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    /* Debug info */
    .debug-info {
        color: #999;
        font-size: 0.8rem;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIMPLE RENDER FUNCTION THAT WILL WORK ---
def render_movie_cards(recommendations, score_column, is_top10_row=False, show_recent=False, show_top10_badge=False):
    # Show count
    st.caption(f"📽️ Found {len(recommendations)} movies")
    
    # Create HTML
    html = '<div class="scroll-container">'
    
    for i, (_, row) in enumerate(recommendations.iterrows()):
        try:
            poster_url, overview, movie_link = fetch_movie_details(row['title'])
            score = row.get(score_column, 85)
            
            if is_top10_row:
                html += f'''
                <div class="top10-wrapper" onclick="window.open(\'{movie_link}\', \'_blank\')">
                    <div class="top10-number">{i+1}</div>
                    <div class="top10-card">
                        <img src="{poster_url}" alt="{row['title']}" onerror="this.src=\'https://via.placeholder.com/300x450?text=No+Image\'">
                        {'<div class="badge-recent">✨ New</div>' if show_recent else ''}
                    </div>
                </div>
                '''
            else:
                html += f'''
                <div class="movie-card" onclick="window.open(\'{movie_link}\', \'_blank\')">
                    <img src="{poster_url}" alt="{row['title']}" onerror="this.src=\'https://via.placeholder.com/280x158?text=No+Image\'">
                    {'<div class="badge-top10">TOP 10</div>' if show_top10_badge else ''}
                    {'<div class="badge-recent">✨ New</div>' if show_recent else ''}
                    <div class="card-match">{score:.0f}% Match</div>
                </div>
                '''
        except:
            continue
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"✅ Displayed {i+1} movies")

# --- SEARCH INPUT ---
search_query = st.text_input("Search movies", placeholder="🔍 Search for movies, shows, genres...", label_visibility="collapsed")

# --- MAIN LOGIC ---
if search_query:
    with st.spinner('Loading...'):
        query_vec = tfidf.transform([search_query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_match_idx = sim_scores.argmax()
        best_score = sim_scores[best_match_idx]
        
        if best_score > 0:
            selected_movie = movies.iloc[best_match_idx]['title']
            poster_url, overview, _ = fetch_movie_details(selected_movie)
            
            # Hero Banner
            hero_html = f"""
            <div class="hero-banner" style="background-image: url('{poster_url}');">
                <div class="hero-vignette"></div>
                <div class="hero-bottom-fade"></div>
                <div class="hero-content">
                    <h1 class="hero-title">{selected_movie}</h1>
                    <p class="hero-meta">{overview}</p>
                    <div class="btn-row">
                        <a href="#" class="btn-play">▶ Play</a>
                        <a href="#" class="btn-info">ⓘ More Info</a>
                    </div>
                </div>
            </div>
            """
            st.markdown(hero_html, unsafe_allow_html=True)
            
            # Content Sections
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            
            st.markdown("""
                <div class="section-header">
                    <div class="section-title">More Like This</div>
                </div>
            """, unsafe_allow_html=True)
            render_movie_cards(get_hybrid_recs(selected_movie), 'Hybrid_Score')
            
            st.markdown("""
                <div class="section-header">
                    <div class="section-title">🔥 Trending Now</div>
                </div>
            """, unsafe_allow_html=True)
            render_movie_cards(get_community_recs(selected_movie), 'CF_Score', show_recent=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
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
                st.info(f"No results found for '{search_query}'")
            st.markdown('</div>', unsafe_allow_html=True)

else:
    # Default Homepage
    hero_image = "https://images.unsplash.com/photo-1626814026160-2237a95fc5a0?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"
    st.markdown(f"""
        <div class="hero-banner" style="background-image: url('{hero_image}');">
            <div class="hero-vignette"></div>
            <div class="hero-bottom-fade"></div>
            <div class="hero-content">
                <h1 class="hero-title">WELCOME TO ZMOVO</h1>
                <p class="hero-meta">Experience the ultimate streaming destination. Watch blockbuster movies, exclusive originals, and trending shows.</p>
                <div class="btn-row">
                    <a href="#" class="btn-play">▶ Start Watching</a>
                    <a href="#" class="btn-info">ⓘ Learn More</a>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Content Sections
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="section-header">
            <div class="section-title">🇲🇾 Top 10 in Malaysia Today</div>
        </div>
    """, unsafe_allow_html=True)
    render_movie_cards(movies.sort_values('vote_count', ascending=False).head(10), 'vote_average', is_top10_row=True, show_recent=True)
    
    st.markdown("""
        <div class="section-header">
            <div class="section-title">🎉 New on Zmovo</div>
        </div>
    """, unsafe_allow_html=True)
    render_movie_cards(movies.head(10), 'vote_average', show_top10_badge=True)
    
    st.markdown("""
        <div class="section-header">
            <div class="section-title">⭐ Popular Movies</div>
        </div>
    """, unsafe_allow_html=True)
    render_movie_cards(movies.sort_values('vote_count', ascending=False).head(10), 'vote_average')
    
    st.markdown('</div>', unsafe_allow_html=True)
