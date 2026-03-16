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

# --- CSS STYLING (Fancy Cinematic Theme) ---
st.markdown("""
    <style>
    /* 1. IMPORT GOOGLE FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;700;900&display=swap');

    /* Force Dark Background and Custom Font */
    .stApp { background-color: #0b0b0c !important; color: #ffffff !important; font-family: 'Montserrat', sans-serif; }
    
    /* Hide Streamlit default headers and footers */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container { padding-top: 0rem !important; }

    /* 2. CSS ANIMATION KEYFRAMES */
    @keyframes slideUpFade { 
        0% { opacity: 0; transform: translateY(40px); } 
        100% { opacity: 1; transform: translateY(0); } 
    }
    @keyframes slowZoom { 
        0% { transform: scale(1); } 
        100% { transform: scale(1.15); } 
    }
    @keyframes pulseGlow { 
        0% { box-shadow: 0 0 0 0 rgba(229, 9, 20, 0.7); } 
        70% { box-shadow: 0 0 0 15px rgba(229, 9, 20, 0); } 
        100% { box-shadow: 0 0 0 0 rgba(229, 9, 20, 0); } 
    }

    /* 3. GLASSMORPHISM NAVBAR */
    .navbar { 
        display: flex; align-items: center; padding: 20px 4%; 
        background: rgba(11, 11, 12, 0.6); /* Semi-transparent */
        backdrop-filter: blur(12px); /* Frosted glass blur */
        -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: -70px; position: relative; z-index: 50; 
        animation: slideUpFade 0.8s ease-out;
    }
    .logo { color: #E50914; font-size: 32px; font-weight: 900; letter-spacing: 2px; margin-right: 40px; text-shadow: 0px 2px 10px rgba(229, 9, 20, 0.5); }
    .nav-links { display: flex; gap: 25px; color: #b3b3b3; font-size: 15px; font-weight: 600; }
    .nav-links span { cursor: pointer; transition: all 0.3s ease; }
    .nav-links span:hover { color: #ffffff; text-shadow: 0px 0px 8px rgba(255,255,255,0.6); transform: translateY(-2px); }

    /* 4. HERO SECTION WITH KEN BURNS EFFECT */
    .hero-container { 
        position: relative; width: 100%; height: 80vh; display: flex; align-items: center; 
        margin-bottom: 30px; overflow: hidden; border-radius: 0 0 20px 20px; background-color: #000;
        animation: slideUpFade 1s ease-out;
    }
    .hero-mask {
        position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        -webkit-mask-image: linear-gradient(to top, transparent 2%, black 40%); 
        mask-image: linear-gradient(to top, transparent 2%, black 40%);
        overflow: hidden;
    }
    .hero-bg { 
        width: 100%; height: 100%; 
        background-size: cover; background-position: center 20%; background-repeat: no-repeat;
        opacity: 0.7; 
        animation: slowZoom 25s infinite alternate linear; /* The magical slow pan/zoom */
    }
    
    /* Hero Content Styling */
    .hero-content { position: relative; z-index: 2; padding: 0 5%; max-width: 65%; animation: slideUpFade 1.5s ease-out; }
    .hero-title { font-size: 4.5rem; font-weight: 900; margin-bottom: 10px; line-height: 1.1; text-transform: uppercase; text-shadow: 3px 3px 6px rgba(0,0,0,0.9); letter-spacing: -1px; }
    .hero-badge { background: linear-gradient(45deg, #E50914, #ff414d); color: white; padding: 6px 12px; border-radius: 4px; font-weight: 700; font-size: 0.9rem; margin-bottom: 20px; display: inline-block; box-shadow: 0 4px 10px rgba(229, 9, 20, 0.4); }
    .hero-desc { font-size: 1.25rem; color: #e5e5e5; text-shadow: 2px 2px 4px rgba(0,0,0,0.9); margin-bottom: 30px; font-weight: 400; line-height: 1.5; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
    
    /* Buttons */
    .hero-buttons button { padding: 12px 28px; font-size: 1.2rem; font-weight: 700; border-radius: 8px; border: none; cursor: pointer; margin-right: 15px; transition: all 0.3s ease; }
    .btn-play { background-color: #ffffff; color: #000000; animation: pulseGlow 2.5s infinite; }
    .btn-play:hover { background-color: #E50914; color: white; transform: scale(1.05); }
    .btn-info { background-color: rgba(109, 109, 110, 0.6); color: white; backdrop-filter: blur(5px); }
    .btn-info:hover { background-color: rgba(255, 255, 255, 0.2); transform: scale(1.05); }

    /* 5. CATEGORIES & ROWS ANIMATION */
    .category-header { font-size: 1.6rem; color: #ffffff; font-weight: 700; margin-top: 40px; margin-bottom: 15px; padding-left: 4%; letter-spacing: 0.5px; animation: slideUpFade 1s ease-out both; }
    .scroll-container { display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 20px; padding: 15px 4% 50px 4%; scroll-behavior: smooth; animation: slideUpFade 1.2s ease-out both; }
    .scroll-container::-webkit-scrollbar { height: 0px; background: transparent; } 
    
    /* 6. NEON GLOW HOVER EFFECTS ON CARDS */
    .movie-card { flex: 0 0 240px; position: relative; transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); cursor: pointer; border-radius: 8px; }
    .movie-card img { width: 100%; aspect-ratio: 2 / 3; object-fit: cover; border-radius: 8px; box-shadow: 0 6px 12px rgba(0,0,0,0.6); transition: all 0.4s ease; border: 2px solid transparent; }
    
    .movie-card:hover { transform: scale(1.08) translateY(-10px); z-index: 10; }
    .movie-card:hover img { border: 2px solid #E50914; box-shadow: 0 15px 30px rgba(229, 9, 20, 0.5); }
    
    /* 7. INTERACTIVE TOP 10 CARDS */
    .top10-card { flex: 0 0 320px; display: flex; align-items: center; position: relative; padding-left: 30px; transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); cursor: pointer; }
    .top10-number { font-size: 280px; font-weight: 900; color: #0b0b0c; -webkit-text-stroke: 4px #444; position: absolute; left: -20px; bottom: -50px; z-index: 1; letter-spacing: -15px; transition: all 0.5s ease; text-shadow: 5px 5px 10px rgba(0,0,0,0.8); }
    .top10-card img { width: 200px; aspect-ratio: 2 / 3; object-fit: cover; border-radius: 8px; z-index: 2; margin-left: 70px; box-shadow: 0 8px 16px rgba(0,0,0,0.8); transition: all 0.4s ease; border: 2px solid transparent; }
    
    .top10-card:hover { transform: translateY(-10px); z-index: 10; }
    .top10-card:hover img { transform: scale(1.1) rotate(2deg); border: 2px solid #E50914; box-shadow: 0 15px 35px rgba(229, 9, 20, 0.6); }
    .top10-card:hover .top10-number { color: rgba(229,9,20,0.1); -webkit-text-stroke: 4px #E50914; transform: scale(1.05) translateX(-10px); text-shadow: 0 0 20px rgba(229,9,20,0.4); }

    /* 8. GLOWING SEARCH BAR */
    .stTextInput { position: relative; z-index: 100 !important; }
    .stTextInput input { color: white !important; background-color: rgba(0,0,0,0.6) !important; border: 1px solid #444 !important; border-radius: 30px !important; padding: 12px 20px !important; font-size: 15px !important; transition: all 0.3s ease !important; }
    .stTextInput input:focus { box-shadow: 0 0 15px rgba(229, 9, 20, 0.6) !important; border-color: #E50914 !important; background-color: rgba(0,0,0,0.8) !important; }
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
        title_safe = str(row['title']).replace('"', '&quot;')
        
        if is_top_10:
            rank = i + 1
            html_content += f'<div class="top10-card"><div class="top10-number">{rank}</div><a href="{movie_link}" target="_blank"><img src="{poster_url}" alt="{title_safe}"></a></div>'
        else:
            html_content += f'<div class="movie-card" title="{title_safe} - {score:.0f}% Match"><a href="{movie_link}" target="_blank"><img src="{poster_url}" alt="{title_safe}"></a></div>'
            
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

# --- SEARCH BAR ---
col1, col2, col3 = st.columns([6, 3, 1])
with col2:
    search_query = st.text_input("", placeholder="🔍 Search titles, characters, genres...", label_visibility="collapsed")

# --- RESULTS SECTION ---
if search_query:
    with st.spinner('Curating cinematic experience...'):
        query_vec = tfidf.transform([search_query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        best_match_idx = sim_scores.argmax()
        best_score = sim_scores[best_match_idx]
        
        if best_score > 0:
            selected_movie = movies.iloc[best_match_idx]['title']
            hero_poster, hero_overview, hero_link = fetch_movie_details(selected_movie)
            
            st.markdown(f"""
                <div class="hero-container">
                    <div class="hero-mask">
                        <div class="hero-bg" style="background-image: url('{hero_poster}');"></div>
                    </div>
                    <div class="hero-content">
                        <div class="hero-title">{selected_movie}</div>
                        <div class="hero-badge">📺 #1 Trending Worldwide</div>
                        <div class="hero-desc">{hero_overview}</div>
                        <div class="hero-buttons">
                            <a href="{hero_link}" target="_blank"><button class="btn-play">▶ Play</button></a>
                            <button class="btn-info">ⓘ More Info</button>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="category-header">Top 10 Movies in Your Area Today</div>', unsafe_allow_html=True)
            render_movie_cards(get_community_recs(selected_movie).head(10), 'CF_Score', is_top_10=True)

            st.markdown(f'<div class="category-header">Because you searched for {selected_movie}</div>', unsafe_allow_html=True)
            render_movie_cards(get_content_based_recs(selected_movie), 'CB_Score')

            st.markdown('<div class="category-header">AI Predictions: You\'ll Love These</div>', unsafe_allow_html=True)
            render_movie_cards(get_hybrid_recs(selected_movie), 'Hybrid_Score')

        else:
            st.warning(f"Searching global TMDB library for topic: '{search_query}'")
            topic_results = search_tmdb_topic(search_query)
            
            if topic_results:
                st.markdown('<div class="category-header">Global Search Results</div>', unsafe_allow_html=True)
                topic_df = pd.DataFrame(topic_results)
                render_movie_cards(topic_df, 'score')
            else:
                st.error("No movies found for that search.")

else:
    st.markdown("""
        <div class="hero-container" style="background: radial-gradient(circle at center, #222 0%, #000 100%);">
            <div class="hero-content" style="text-align: center; margin: 0 auto; max-width: 100%;">
                <div class="hero-title" style="color:#555; text-shadow: none; font-size: 3rem;">FIND YOUR NEXT OBSESSION</div>
                <div class="hero-desc" style="color:#888;">Type a movie title or mood in the top right to unleash the recommendation engine.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
