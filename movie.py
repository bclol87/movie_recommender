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

# --- CSS STYLING (Ultra-Premium Netflix Theme) ---
st.markdown("""
    <style>
    /* 1. IMPORT GOOGLE FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;800;900&display=swap');

    /* Force Dark Background and Custom Font */
    .stApp { background-color: #141414 !important; color: #ffffff !important; font-family: 'Montserrat', sans-serif; overflow-x: hidden; }
    
    /* Hide Streamlit default headers and footers */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; max-width: 100% !important; padding-left: 0 !important; padding-right: 0 !important;}

    /* 2. CSS ANIMATION KEYFRAMES */
    @keyframes slideUpFade { 
        0% { opacity: 0; transform: translateY(40px); } 
        100% { opacity: 1; transform: translateY(0); } 
    }
    @keyframes floatPoster {
        0% { transform: translateY(0px); box-shadow: 0 25px 50px rgba(0,0,0,0.8); }
        50% { transform: translateY(-15px); box-shadow: 0 35px 60px rgba(0,0,0,0.9); }
        100% { transform: translateY(0px); box-shadow: 0 25px 50px rgba(0,0,0,0.8); }
    }
    @keyframes ambientPan {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 3. GLASSMORPHISM NAVBAR */
    .navbar { 
        display: flex; align-items: center; padding: 20px 5%; 
        background: linear-gradient(to bottom, rgba(20,20,20,0.9) 0%, rgba(20,20,20,0) 100%);
        margin-bottom: -80px; position: relative; z-index: 50; 
        animation: slideUpFade 0.8s ease-out;
    }
    .logo { color: #E50914; font-size: 34px; font-weight: 900; letter-spacing: 1px; margin-right: 50px; text-shadow: 0px 2px 10px rgba(229, 9, 20, 0.4); }
    .nav-links { display: flex; gap: 30px; color: #e5e5e5; font-size: 15px; font-weight: 500; }
    .nav-links span { cursor: pointer; transition: color 0.3s ease; }
    .nav-links span:hover { color: #ffffff; font-weight: 600; }

    /* 4. DUAL-LAYER HERO SECTION (Fixes the stretched image issue) */
    .hero-container { 
        position: relative; width: 100%; height: 85vh; display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 20px; overflow: hidden; background-color: #000;
        animation: slideUpFade 1s ease-out;
    }
    
    /* Layer 1: The Ambient Blurred Background */
    .hero-bg { 
        position: absolute; top: -10%; left: -10%; width: 120%; height: 120%; 
        background-size: cover; background-position: center; 
        filter: blur(40px) brightness(0.35); /* Super blurred and dark */
        z-index: 0; animation: ambientPan 30s ease-in-out infinite;
    }
    
    /* Layer 2: The Vignette Gradient (Makes text readable) */
    .hero-vignette {
        position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1;
        background: radial-gradient(circle at center, transparent 0%, rgba(20,20,20,0.8) 100%),
                    linear-gradient(to right, rgba(20,20,20,1) 0%, transparent 50%),
                    linear-gradient(to top, rgba(20,20,20,1) 0%, transparent 20%);
    }
    
    /* Layer 3: Left Side Content */
    .hero-content { position: relative; z-index: 2; padding-left: 5%; width: 55%; animation: slideUpFade 1.2s ease-out; }
    .hero-title { font-size: 5rem; font-weight: 900; margin-bottom: 15px; line-height: 1.05; text-transform: uppercase; text-shadow: 2px 4px 10px rgba(0,0,0,0.8); letter-spacing: -2px; }
    .hero-badge { background-color: #E50914; color: white; padding: 6px 14px; border-radius: 4px; font-weight: 800; font-size: 0.95rem; margin-bottom: 25px; display: inline-block; box-shadow: 0 4px 15px rgba(229, 9, 20, 0.5); }
    .hero-desc { font-size: 1.3rem; color: #d2d2d2; text-shadow: 1px 2px 5px rgba(0,0,0,0.9); margin-bottom: 35px; font-weight: 400; line-height: 1.5; display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; max-width: 90%; }
    
    /* Buttons */
    .hero-buttons button { padding: 14px 32px; font-size: 1.3rem; font-weight: 800; border-radius: 6px; border: none; cursor: pointer; margin-right: 20px; transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); display: flex; align-items: center; justify-content: center; gap: 10px; display: inline-flex;}
    .btn-play { background-color: #ffffff; color: #000000; }
    .btn-play:hover { background-color: #d8d8d8; transform: scale(1.05); }
    .btn-info { background-color: rgba(109, 109, 110, 0.7); color: white; backdrop-filter: blur(10px); }
    .btn-info:hover { background-color: rgba(109, 109, 110, 0.4); transform: scale(1.05); }

    /* Layer 4: Right Side Floating Poster */
    .hero-poster-wrapper { position: relative; z-index: 2; width: 45%; display: flex; justify-content: center; align-items: center; padding-right: 5%; animation: slideUpFade 1.5s ease-out; }
    .hero-poster-wrapper img { height: 65vh; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); animation: floatPoster 6s ease-in-out infinite; }

    /* 5. CAROUSEL ROWS */
    .content-wrapper { padding: 0 5%; }
    .category-header { font-size: 1.5rem; color: #e5e5e5; font-weight: 600; margin-top: 30px; margin-bottom: 15px; animation: slideUpFade 1s ease-out both; }
    .scroll-container { display: flex; flex-wrap: nowrap; overflow-x: auto; overflow-y: visible; gap: 15px; padding: 20px 0 60px 0; scroll-behavior: smooth; animation: slideUpFade 1.2s ease-out both; }
    .scroll-container::-webkit-scrollbar { height: 0px; background: transparent; } 
    
    /* 6. NETFLIX POP-OUT HOVER EFFECT */
    .movie-card { flex: 0 0 220px; position: relative; transition: transform 0.4s cubic-bezier(0.25, 1, 0.5, 1), z-index 0s; cursor: pointer; border-radius: 6px; }
    .movie-card img { width: 100%; aspect-ratio: 2 / 3; object-fit: cover; border-radius: 6px; box-shadow: 0 4px 8px rgba(0,0,0,0.6); transition: all 0.4s ease; border: 2px solid transparent; }
    
    .movie-card:hover { transform: scale(1.15) translateY(-15px); z-index: 99; transition-delay: 0.1s; }
    .movie-card:hover img { border: 2px solid #ffffff; box-shadow: 0 20px 40px rgba(0,0,0,0.9); border-radius: 8px;}
    
    /* 7. TOP 10 RANKING CARDS */
    .top10-card { flex: 0 0 300px; display: flex; align-items: center; position: relative; padding-left: 20px; transition: transform 0.4s cubic-bezier(0.25, 1, 0.5, 1); cursor: pointer; }
    .top10-number { font-size: 250px; font-weight: 900; color: #141414; -webkit-text-stroke: 4px #595959; position: absolute; left: -25px; bottom: -40px; z-index: 1; letter-spacing: -15px; transition: all 0.4s ease; }
    .top10-card img { width: 170px; aspect-ratio: 2 / 3; object-fit: cover; border-radius: 6px; z-index: 2; margin-left: 60px; box-shadow: 0 8px 16px rgba(0,0,0,0.8); transition: all 0.4s ease; border: 2px solid transparent;}
    
    .top10-card:hover { transform: scale(1.1) translateY(-10px); z-index: 99; transition-delay: 0.1s; }
    .top10-card:hover img { border: 2px solid #ffffff; box-shadow: 0 20px 40px rgba(0,0,0,0.9); }
    .top10-card:hover .top10-number { -webkit-text-stroke: 4px #ffffff; color: rgba(255,255,255,0.1); }

    /* 8. SEARCH BAR OVERRIDE */
    .stTextInput { position: absolute; top: 25px; right: 5%; width: 300px !important; z-index: 100 !important; }
    .stTextInput input { color: white !important; background-color: rgba(0,0,0,0.7) !important; border: 1px solid #fff !important; padding: 10px 15px !important; font-size: 14px !important; transition: all 0.3s ease !important; }
    .stTextInput input:focus { background-color: #141414 !important; width: 350px !important; }
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

# --- SEARCH BAR (Now floating perfectly via absolute CSS) ---
search_query = st.text_input("", placeholder="🔍 Search titles, characters, genres...", label_visibility="collapsed")

# --- HELPER FUNCTION TO RENDER UI CARDS ---
def render_movie_cards(recommendations, score_column, is_top_10=False):
    html_content = '<div class="content-wrapper"><div class="scroll-container">'
    for i, (_, row) in enumerate(recommendations.iterrows()):
        poster_url, overview, movie_link = fetch_movie_details(row['title'])
        title_safe = str(row['title']).replace('"', '&quot;')
        
        if is_top_10:
            rank = i + 1
            html_content += f'<div class="top10-card"><div class="top10-number">{rank}</div><a href="{movie_link}" target="_blank"><img src="{poster_url}" alt="{title_safe}"></a></div>'
        else:
            html_content += f'<div class="movie-card" title="{title_safe}"><a href="{movie_link}" target="_blank"><img src="{poster_url}" alt="{title_safe}"></a></div>'
            
    html_content += '</div></div>'
    st.markdown(html_content, unsafe_allow_html=True)

# --- RESULTS SECTION ---
if search_query:
    with st.spinner('Loading cinematic experience...'):
        query_vec = tfidf.transform([search_query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        best_match_idx = sim_scores.argmax()
        best_score = sim_scores[best_match_idx]
        
        if best_score > 0:
            selected_movie = movies.iloc[best_match_idx]['title']
            hero_poster, hero_overview, hero_link = fetch_movie_details(selected_movie)
            
            # --- THE DUAL LAYER HERO FIX ---
            st.markdown(f"""
                <div class="hero-container">
                    <div class="hero-bg" style="background-image: url('{hero_poster}');"></div>
                    <div class="hero-vignette"></div>
                    
                    <div class="hero-content">
                        <div class="hero-title">{selected_movie}</div>
                        <div class="hero-badge">TOP 10 TODAY</div>
                        <div class="hero-desc">{hero_overview}</div>
                        <div class="hero-buttons">
                            <a href="{hero_link}" target="_blank" style="text-decoration:none;">
                                <button class="btn-play">
                                    <svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M8 5v14l11-7z"/></svg> 
                                    Play
                                </button>
                            </a>
                            <button class="btn-info">
                                <svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>
                                More Info
                            </button>
                        </div>
                    </div>
                    
                    <div class="hero-poster-wrapper">
                        <img src="{hero_poster}" alt="{selected_movie}">
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="content-wrapper"><div class="category-header">Top 10 Movies in Your Area Today</div></div>', unsafe_allow_html=True)
            render_movie_cards(get_community_recs(selected_movie).head(10), 'CF_Score', is_top_10=True)

            st.markdown(f'<div class="content-wrapper"><div class="category-header">Because you searched for {selected_movie}</div></div>', unsafe_allow_html=True)
            render_movie_cards(get_content_based_recs(selected_movie), 'CB_Score')

            st.markdown('<div class="content-wrapper"><div class="category-header">AI Predictions: You\'ll Love These</div></div>', unsafe_allow_html=True)
            render_movie_cards(get_hybrid_recs(selected_movie), 'Hybrid_Score')

        else:
            st.warning(f"Searching global TMDB library for topic: '{search_query}'")
            topic_results = search_tmdb_topic(search_query)
            if topic_results:
                st.markdown('<div class="content-wrapper"><div class="category-header">Global Search Results</div></div>', unsafe_allow_html=True)
                topic_df = pd.DataFrame(topic_results)
                render_movie_cards(topic_df, 'score')
            else:
                st.error("No movies found for that search.")

else:
    # Default State
    st.markdown("""
        <div class="hero-container">
             <div class="hero-bg" style="background-image: url('https://assets.nflxext.com/ffe/siteui/vlv3/1ecf18b2-adad-4684-bd9a-acab7f2a875f/728df0cc-b789-4bba-9ea7-626a5c2d36ab/MY-en-20230116-popsignuptwoweeks-perspective_alpha_website_medium.jpg'); opacity: 0.5; filter: blur(5px);"></div>
             <div class="hero-vignette"></div>
             <div class="hero-content" style="width: 100%; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%;">
                 <div class="hero-title" style="font-size: 4rem;">Unlimited movies, TV shows, and more.</div>
                 <div class="hero-desc" style="font-size: 1.5rem; text-align: center;">Search a movie to trigger the AI Recommendation Engine.</div>
             </div>
        </div>
    """, unsafe_allow_html=True)
