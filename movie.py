import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- Import logic from movie_logic.py ---
from movie_logic import (
    movies, cosine_sim, tfidf, tfidf_matrix,
    fetch_movie_details, search_tmdb_topic,
    get_content_based_recs, get_community_recs, get_hybrid_recs
)

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="Zmovo Pro", page_icon="▶️", layout="wide")

# Deep Blue/Cyan Theme CSS
st.markdown("""
    <style>
    /* Main Background & Text */
    .stApp { background-color: #070b19 !important; color: #ffffff !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Hide default Streamlit headers/footers */
    .block-container { padding-top: 1rem !important; max-width: 95% !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

    /* Top Navigation Mockup */
    .top-nav { display: flex; justify-content: space-between; align-items: center; padding: 10px 0 30px 0; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;}
    .nav-logo { font-size: 1.8rem; font-weight: 800; color: #ffffff; display: flex; align-items: center; gap: 8px;}
    .nav-logo span { color: #f39c12; }
    .nav-logo i { color: #00d2ff; }
    .nav-links { display: flex; gap: 20px; font-size: 0.95rem; color: #a0a5b5; font-weight: 500; }
    .nav-links span:hover { color: #00d2ff; cursor: pointer; }
    .login-btn { background-color: #00d2ff; color: #000; padding: 8px 20px; border-radius: 4px; font-weight: 600; text-decoration: none; font-size: 0.9rem;}

    /* Left Sidebar Menu */
    .left-menu { background-color: #0b1121; border-radius: 12px; padding: 15px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    .menu-btn { background-color: #00d2ff; color: #000 !important; font-weight: 700; border-radius: 6px; padding: 12px 20px; margin: 0 15px 15px 15px; text-align: center; display: block; text-decoration: none;}
    .menu-item { padding: 12px 20px; color: #a0a5b5; font-size: 0.95rem; font-weight: 500; display: flex; justify-content: space-between; align-items: center; cursor: pointer; border-left: 3px solid transparent;}
    .menu-item:hover { background-color: rgba(255,255,255,0.03); color: #fff; }
    .menu-item.active { color: #00d2ff; border-left: 3px solid #00d2ff; background-color: rgba(0, 210, 255, 0.05); }

    /* Hero Banner */
    .hero-banner { 
        background: linear-gradient(to right, #0b2239 10%, rgba(11, 34, 57, 0.4) 100%), url('https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80');
        background-size: cover; background-position: center; border-radius: 12px; padding: 50px 40px; margin-bottom: 30px; display: flex; flex-direction: column; justify-content: center; min-height: 350px; box-shadow: 0 10px 20px rgba(0,0,0,0.4);
    }
    .hero-badge { background-color: #f39c12; color: #fff; font-size: 0.7rem; font-weight: 800; padding: 4px 8px; border-radius: 3px; letter-spacing: 1px; display: inline-block; margin-right: 10px;}
    .hero-sub { color: #a0a5b5; font-size: 0.9rem; font-weight: 500; }
    .hero-title { font-size: 2.8rem; font-weight: 800; margin: 15px 0; color: #fff; }
    .hero-meta { color: #d1d5e0; font-size: 0.9rem; margin-bottom: 25px; line-height: 1.6; }
    .play-trailer-btn { display: inline-flex; align-items: center; gap: 10px; color: #fff; text-decoration: none; font-weight: 600; font-size: 0.9rem;}
    .play-trailer-btn i { border: 2px solid #fff; border-radius: 50%; width: 30px; height: 30px; display: flex; justify-content: center; align-items: center; font-style: normal;}
    .play-trailer-btn:hover i { border-color: #00d2ff; color: #00d2ff; }

    /* Section Headers */
    .section-title { font-size: 1.1rem; font-weight: 700; color: #fff; margin-bottom: 15px; border-left: 4px solid #fff; padding-left: 10px; text-transform: uppercase; letter-spacing: 1px;}

    /* Scroll Container & Cards */
    .scroll-container { display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 18px; padding-bottom: 20px; }
    .scroll-container::-webkit-scrollbar { height: 6px; }
    .scroll-container::-webkit-scrollbar-track { background: transparent; }
    .scroll-container::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 10px; }
    
    .z-card { flex: 0 0 180px; background-color: #0b1121; border-radius: 8px; overflow: hidden; position: relative; cursor: pointer; transition: transform 0.2s; box-shadow: 0 4px 15px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.02);}
    .z-card:hover { transform: translateY(-5px); }
    
    .img-wrapper { position: relative; height: 260px; }
    .img-wrapper img { width: 100%; height: 100%; object-fit: cover; }
    .img-overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(to top, rgba(11,17,33,1) 0%, rgba(11,17,33,0) 50%); }
    
    .card-content { padding: 12px; position: absolute; bottom: 0; width: 100%; }
    .card-title { font-size: 0.9rem; font-weight: 700; color: #fff; margin-bottom: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .card-genre { color: #f39c12; font-size: 0.75rem; font-weight: 500; margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
    .card-footer { display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; color: #a0a5b5; }
    .card-footer i { color: #e74c3c; font-style: normal; font-size: 1rem;}

    /* Streamlit overrides */
    div[data-testid="stTextInput"] input { background-color: #0b1121 !important; color: white !important; border: 1px solid #1e293b !important; border-radius: 20px !important; padding: 10px 20px !important;}
    div[data-testid="stTextInput"] input:focus { border-color: #00d2ff !important; box-shadow: 0 0 0 1px #00d2ff !important;}
    </style>
""", unsafe_allow_html=True)

# --- TOP NAVIGATION BAR ---
st.markdown("""
    <div class="top-nav">
        <div class="nav-logo"><i>▶</i> Zmovo<span>.</span></div>
        <div class="nav-links">
            <span>Browse ⌄</span> <span>Movies ⌄</span> <span>TV Series ⌄</span> <span>Pages</span> <span>Premium</span>
        </div>
        <div>
            <a href="#" class="login-btn">👤 Login</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- HELPER FUNCTION TO RENDER UI CARDS ---
# --- HELPER FUNCTION TO RENDER UI CARDS ---
def render_movie_cards(recommendations, score_column):
    html_content = '<div class="scroll-container">'
    
    for i, (_, row) in enumerate(recommendations.iterrows()):
        poster_url, overview, movie_link = fetch_movie_details(row['title'])
        score = row.get(score_column, 85)
        
        # Calculate a mock duration based on the score for the UI
        hours = int(score) % 2 + 1
        mins = int(score) % 60
        
        # NOTE: The HTML below has no leading spaces so Streamlit doesn't treat it as a code block!
        html_content += f"""<div class="z-card" onclick="window.open('{movie_link}', '_blank')">
<div class="img-wrapper">
<img src="{poster_url}" alt="{row['title']}">
<div class="img-overlay"></div>
</div>
<div class="card-content">
<div class="card-title">{row['title']}</div>
<div class="card-genre">Match Score: {score:.0f}%</div>
<div class="card-footer">
<span>⏱ {hours} Hr {mins} Min</span>
<span>♡</span>
</div>
</div>
</div>"""
        
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

# --- MAIN LAYOUT (LEFT MENU + RIGHT CONTENT) ---
menu_col, content_col = st.columns([1, 4], gap="large")

# -- Left Column: Menu --
with menu_col:
    st.markdown("""
        <div class="left-menu">
            <a href="#" class="menu-btn">TOP 10 MOVIES</a>
            <div class="menu-item">Home <span>🏠</span></div>
            <div class="menu-item active">Romantic Movies <span>💙</span></div>
            <div class="menu-item">Top 10 Movies <span>🏅</span></div>
            <div class="menu-item">Rating Movies <span>📊</span></div>
            <div class="menu-item">New Movies <span>🎬</span></div>
            <div class="menu-item">2024 All Movies <span>✓</span></div>
            <div class="menu-item">TV Series <span>📺</span></div>
        </div>
    """, unsafe_allow_html=True)

# -- Right Column: Content --
with content_col:
    
    # Search Bar natively integrated
    search_query = st.text_input("Search", placeholder="🔍 Search for movies, actors, or genres...", label_visibility="collapsed")
    
    if search_query:
        with st.spinner('Loading your selections...'):
            query_vec = tfidf.transform([search_query])
            sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            best_match_idx = sim_scores.argmax()
            best_score = sim_scores[best_match_idx]
            
            if best_score > 0:
                selected_movie = movies.iloc[best_match_idx]['title']
                
                # Fetch details for the HERO BANNER dynamically based on the top result
                poster_url, overview, _ = fetch_movie_details(selected_movie)
                genres = movies.iloc[best_match_idx]['genres_clean'].replace(" ", ", ")
                
                # Dynamically inject the background image of the hero banner
                hero_html = f"""
                <div class="hero-banner" style="background: linear-gradient(to right, #0b2239 10%, rgba(11, 34, 57, 0.6) 100%), url('{poster_url}'); background-size: cover; background-position: center 20%;">
                    <div>
                        <span class="hero-badge">PREMIUM</span> <span class="hero-sub">AI Recommended</span>
                        <h1 class="hero-title">{selected_movie}</h1>
                        <p class="hero-meta">Category : Global Movies<br>Genre : {genres}</p>
                        <a href="#" class="play-trailer-btn"><i>▶</i> PLAY TRAILER</a>
                    </div>
                </div>
                """
                st.markdown(hero_html, unsafe_allow_html=True)
                
                # Sections
                st.markdown('<div class="section-title">✨ NEW ARRIVALS (HYBRID PICKS)</div>', unsafe_allow_html=True)
                render_movie_cards(get_hybrid_recs(selected_movie), 'Hybrid_Score')
                
                st.markdown('<br><div class="section-title">🔥 TRENDING NOW (COMMUNITY)</div>', unsafe_allow_html=True)
                render_movie_cards(get_community_recs(selected_movie), 'CF_Score')
                
            else:
                st.markdown('<div class="section-title">🌐 GLOBAL SEARCH RESULTS</div>', unsafe_allow_html=True)
                topic_results = search_tmdb_topic(search_query)
                if topic_results:
                    render_movie_cards(pd.DataFrame(topic_results), 'score')
                else:
                    st.warning("No movies found.")
    else:
        # Static Hero Banner for when the page first loads (No search query yet)
        st.markdown("""
            <div class="hero-banner">
                <div>
                    <span class="hero-badge">PREMIUM</span> <span class="hero-sub">Period Adventure</span>
                    <h1 class="hero-title">Welcome to Zmovo</h1>
                    <p class="hero-meta">Category : All Movies<br>Search above to start discovering.</p>
                    <a href="#" class="play-trailer-btn"><i>▶</i> BROWSE NOW</a>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Show some default community picks when empty
        st.markdown('<div class="section-title">I POPULAR MOVIES</div>', unsafe_allow_html=True)
        render_movie_cards(movies.sort_values('vote_count', ascending=False).head(10), 'vote_average')
