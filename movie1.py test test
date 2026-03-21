import streamlit as st
import streamlit.components.v1 as components
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

# --- CSS STYLING (Ultra-Premium Cinematic Theme) ---
st.markdown("""
<style>
/* 1. IMPORT GOOGLE FONTS */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;700;900&display=swap');

/* Force Dark Background and Custom Font */
.stApp { background-color: #0b0b0c !important; color: #ffffff !important; font-family: 'Montserrat', sans-serif; overflow-x: hidden; }

/* Hide Streamlit default headers and footers */
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; max-width: 100% !important; padding-left: 0 !important; padding-right: 0 !important;}

/* 2. CSS ANIMATION KEYFRAMES */
@keyframes slideUpFade { 
    0% { opacity: 0; transform: translateY(40px); } 
    100% { opacity: 1; transform: translateY(0); } 
}
@keyframes floatPoster {
    0% { transform: translateY(0px); box-shadow: 0 15px 35px rgba(0,0,0,0.9); }
    50% { transform: translateY(-15px); box-shadow: 0 25px 50px rgba(229, 9, 20, 0.2); }
    100% { transform: translateY(0px); box-shadow: 0 15px 35px rgba(0,0,0,0.9); }
}

/* 3. ULTRA-PREMIUM MINIMALIST NAVBAR */
.navbar { 
    display: flex; align-items: center; justify-content: space-between; padding: 25px 5%; 
    background: linear-gradient(to bottom, rgba(0, 0, 0, 0.9) 0%, rgba(0, 0, 0, 0.4) 50%, rgba(0, 0, 0, 0) 100%);
    margin-bottom: -100px; position: relative; z-index: 50; 
    animation: slideUpFade 0.8s ease-out;
    pointer-events: none; /* Allows clicks to pass through empty space */
}
.logo { 
    color: #E50914; 
    font-size: 38px; 
    font-weight: 900; 
    letter-spacing: 2px; 
    text-transform: uppercase;
    text-shadow: 0px 0px 20px rgba(229, 9, 20, 0.6); 
    transition: transform 0.3s ease, text-shadow 0.3s ease;
    cursor: pointer;
    pointer-events: auto;
}
.logo:hover {
    transform: scale(1.05);
    text-shadow: 0px 0px 25px rgba(229, 9, 20, 1);
}

/* 4. DUAL-LAYER HERO SECTION */
.hero-container { 
    position: relative; width: 100%; height: 85vh; display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 30px; overflow: hidden; background-color: #0b0b0c;
    border-bottom: 1px solid #1a1a1a;
    animation: slideUpFade 1s ease-out;
}

/* Layer 1: Ambient Blurred Background Glow */
.hero-bg-glow { 
    position: absolute; top: -10%; left: -10%; width: 120%; height: 120%; 
    background-size: cover; background-position: center; background-repeat: no-repeat;
    filter: blur(45px) brightness(0.35); 
    z-index: 0;
}

/* Layer 2: Left Side Text Content */
.hero-content { position: relative; z-index: 2; padding-left: 5%; width: 55%; margin-top: 40px; animation: slideUpFade 1.5s ease-out; }
.hero-title { font-size: 4.8rem; font-weight: 900; margin-bottom: 10px; line-height: 1.1; text-transform: uppercase; text-shadow: 3px 3px 8px rgba(0,0,0,0.9); letter-spacing: -1px; }
.hero-badge { background: linear-gradient(45deg, #E50914, #ff414d); color: white; padding: 6px 14px; border-radius: 4px; font-weight: 800; font-size: 0.9rem; margin-bottom: 25px; display: inline-block; box-shadow: 0 4px 10px rgba(229, 9, 20, 0.4); text-transform: uppercase; letter-spacing: 1px; }
.hero-desc { font-size: 1.25rem; color: #e5e5e5; text-shadow: 2px 2px 5px rgba(0,0,0,0.9); margin-bottom: 35px; font-weight: 400; line-height: 1.6; display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; max-width: 90%; }

/* Buttons */
.hero-buttons button { padding: 14px 30px; font-size: 1.2rem; font-weight: 700; border-radius: 6px; border: none; cursor: pointer; margin-right: 15px; transition: all 0.3s ease; display: inline-flex; align-items: center; gap: 8px;}
.btn-play { background-color: #ffffff; color: #000000; }
.btn-play:hover { background-color: #E50914; color: white; transform: scale(1.05); }
.btn-info { background-color: rgba(109, 109, 110, 0.6); color: white; backdrop-filter: blur(8px); }
.btn-info:hover { background-color: rgba(255, 255, 255, 0.25); transform: scale(1.05); }

/* Layer 3: Right Side Clear Poster - ADDED MARGIN TO PUSH DOWN FROM SEARCH BAR */
.hero-poster-box {
    position: relative; z-index: 2; width: 45%; display: flex; justify-content: center; align-items: center;
    padding-right: 5%; margin-top: 60px; animation: slideUpFade 1.2s ease-out;
}
.hero-poster-box img {
    height: 55vh; /* Reduced height so it fits beautifully without hitting the top */
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.15);
    animation: floatPoster 6s ease-in-out infinite; 
}

/* 5. CATEGORIES & ROWS ANIMATION */
.category-header { font-size: 1.6rem; color: #ffffff; font-weight: 700; margin-top: 40px; margin-bottom: 15px; padding-left: 4%; letter-spacing: 0.5px; animation: slideUpFade 1s ease-out both; }
.scroll-container { 
    display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 20px; 
    padding: 15px 4% 50px 4%; scroll-behavior: smooth; animation: slideUpFade 1.2s ease-out both; 
    scroll-snap-type: x mandatory;
    -webkit-overflow-scrolling: touch;
}
.scroll-container::-webkit-scrollbar { height: 8px; background: rgba(255,255,255,0.02); border-radius: 10px;} 
.scroll-container::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 10px;}
.scroll-container::-webkit-scrollbar-thumb:hover { background: #E50914; }

/* 6. NEON GLOW HOVER EFFECTS ON CARDS */
.movie-card { flex: 0 0 240px; position: relative; transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); cursor: pointer; border-radius: 8px; scroll-snap-align: start; scroll-margin-left: 4%;}
.movie-card img { width: 100%; aspect-ratio: 2 / 3; object-fit: cover; border-radius: 8px; box-shadow: 0 6px 12px rgba(0,0,0,0.6); transition: all 0.4s ease; border: 2px solid transparent; }
.movie-card:hover { transform: scale(1.08) translateY(-10px); z-index: 10; }
.movie-card:hover img { border: 2px solid #E50914; box-shadow: 0 15px 30px rgba(229, 9, 20, 0.5); }

/* 7. INTERACTIVE TOP 10 CARDS */
.top10-card { flex: 0 0 320px; display: flex; align-items: center; position: relative; padding-left: 30px; transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); cursor: pointer; scroll-snap-align: start; scroll-margin-left: 4%;}
.top10-number { font-size: 280px; font-weight: 900; color: #0b0b0c; -webkit-text-stroke: 4px #444; position: absolute; left: -20px; bottom: -50px; z-index: 1; letter-spacing: -15px; transition: all 0.5s ease; text-shadow: 5px 5px 10px rgba(0,0,0,0.8); }
.top10-card img { width: 200px; aspect-ratio: 2 / 3; object-fit: cover; border-radius: 8px; z-index: 2; margin-left: 70px; box-shadow: 0 8px 16px rgba(0,0,0,0.8); transition: all 0.4s ease; border: 2px solid transparent; }
.top10-card:hover { transform: translateY(-10px); z-index: 10; }
.top10-card:hover img { transform: scale(1.1) rotate(2deg); border: 2px solid #E50914; box-shadow: 0 15px 35px rgba(229, 9, 20, 0.6); }
.top10-card:hover .top10-number { color: rgba(229,9,20,0.1); -webkit-text-stroke: 4px #E50914; transform: scale(1.05) translateX(-10px); text-shadow: 0 0 20px rgba(229,9,20,0.4); }

/* 8. SLEEK CINEMATIC SEARCH BAR (NETFLIX STYLE) */
.stTextInput { position: absolute; top: 15px; right: 5%; width: 280px !important; z-index: 100 !important; pointer-events: auto;}
.stTextInput input { 
    color: white !important; 
    background-color: rgba(0, 0, 0, 0.6) !important; /* Dark, sleek background */
    border: 1px solid rgba(255, 255, 255, 0.2) !important; /* Subtle, classy outline */
    border-radius: 6px !important; /* Authentic streaming app square-round edges */
    padding: 12px 20px !important; 
    font-size: 15px !important; 
    transition: all 0.3s ease !important; 
}
.stTextInput input::placeholder { color: rgba(255,255,255,0.4) !important; font-weight: 300 !important; }
.stTextInput input:focus { 
    background-color: rgba(0, 0, 0, 0.9) !important; 
    border-color: #E50914 !important; /* Sharp Netflix red border on click */
    box-shadow: none !important; /* Removed messy outer glow for a cleaner look */
    width: 320px !important; 
}
</style>
""", unsafe_allow_html=True)

# --- MAGIC JAVASCRIPT INJECTION: Target ONLY the Top 10 Row ---
components.html(
    """
    <script>
    const doc = window.parent.document;
    if (!doc.getElementById('h-scroll-script')) {
        const script = doc.createElement('script');
        script.id = 'h-scroll-script';
        script.innerHTML = `
            document.addEventListener('wheel', function(e) {
                const container = e.target.closest('.top10-scroll-row');
                if (container) {
                    const atLeft = container.scrollLeft === 0 && e.deltaY < 0;
                    const atRight = container.scrollLeft >= (container.scrollWidth - container.clientWidth - 2) && e.deltaY > 0;
                    
                    if (!atLeft && !atRight && e.deltaY !== 0) {
                        e.preventDefault();
                        container.scrollLeft += e.deltaY * 2.5; 
                    }
                }
            }, { passive: false });
        `;
        doc.head.appendChild(script);
    }
    </script>
    """,
    height=0, width=0
)

# --- NAVIGATION BAR (Super Clean & Minimalist) ---
st.markdown("""
<div class="navbar">
<div class="logo">CineMatch</div>
</div>
""", unsafe_allow_html=True)

# --- SEARCH BAR ---
search_query = st.text_input("", placeholder="🔍 Search titles, genres, actors...", label_visibility="collapsed")

# --- HELPER FUNCTION TO RENDER UI CARDS ---
def render_movie_cards(recommendations, score_column, is_top_10=False):
    container_class = "scroll-container top10-scroll-row" if is_top_10 else "scroll-container"
    html_content = f'<div class="{container_class}">'
    
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
            
            # COMPLETELY FLATTENED HTML to prevent Streamlit code blocks
            st.markdown(f"""
<div class="hero-container">
<div class="hero-bg-glow" style="background-image: url('{hero_poster}');"></div>
<div class="hero-content">
<div class="hero-title">{selected_movie}</div>
<div class="hero-badge">#1 Trending Worldwide</div>
<div class="hero-desc">{hero_overview}</div>
<div class="hero-buttons">
<a href="{hero_link}" target="_blank" style="text-decoration:none;">
<button class="btn-play">▶ Play</button>
</a>
<button class="btn-info">ⓘ More Info</button>
</div>
</div>
<div class="hero-poster-box">
<img src="{hero_poster}" alt="{selected_movie}">
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
    # COMPLETELY FLATTENED HTML to prevent Streamlit code blocks
    st.markdown("""
<div class="hero-container">
<div class="hero-bg-glow" style="background-image: url('https://assets.nflxext.com/ffe/siteui/vlv3/1ecf18b2-adad-4684-bd9a-acab7f2a875f/728df0cc-b789-4bba-9ea7-626a5c2d36ab/MY-en-20230116-popsignuptwoweeks-perspective_alpha_website_medium.jpg'); opacity: 0.5;"></div>
<div class="hero-content" style="width: 100%; text-align: center; padding: 0; margin-top: 0;">
<div class="hero-title" style="font-size: 3.5rem; text-shadow: 2px 4px 10px rgba(0,0,0,0.8);">FIND YOUR NEXT OBSESSION</div>
<div class="hero-desc" style="color:#ddd; font-size: 1.4rem;">Type a movie title or mood in the top right to unleash the recommendation engine.</div>
</div>
</div>
""", unsafe_allow_html=True)
