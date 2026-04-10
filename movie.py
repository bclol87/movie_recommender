import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import urllib.parse
import re 

# --- MAGIC STEP: Import our functions from our new logic file ---
from movie_logic import (
    movies, cosine_sim, tfidf, tfidf_matrix,
    fetch_movie_details, search_tmdb_topic,
    get_content_based_recs, get_community_recs, get_hybrid_recs,
    get_profile_based_recs 
)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Tarflix Pro", page_icon="🍿", layout="wide")

# --- SESSION STATE INITIALIZATION (MEMORY BANK) ---
if 'liked_movies' not in st.session_state:
    st.session_state.liked_movies = []

# --- MULTI-LIKE MEMORY FIX ---
if "likes" in st.query_params:
    likes_str = urllib.parse.unquote(st.query_params["likes"])
    if likes_str:
        st.session_state.liked_movies = likes_str.split("|") 
    else:
        st.session_state.liked_movies = []
    
    del st.query_params["likes"] 
    st.rerun() 

if "q" in st.query_params and "search_query" not in st.session_state:
    st.session_state.search_query = st.query_params["q"]

# --- CSS STYLING (Ultra-Premium Cinematic Theme) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;700;900&display=swap');
.stApp { background-color: #0b0b0c !important; color: #ffffff !important; font-family: 'Montserrat', sans-serif; overflow-x: hidden; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; max-width: 100% !important; padding-left: 0 !important; padding-right: 0 !important;}

@keyframes slideUpFade { 0% { opacity: 0; transform: translateY(40px); } 100% { opacity: 1; transform: translateY(0); } }
@keyframes floatPoster {
    0% { transform: translateY(0px); box-shadow: 0 15px 35px rgba(0,0,0,0.9); }
    50% { transform: translateY(-15px); box-shadow: 0 25px 50px rgba(229, 9, 20, 0.2); }
    100% { transform: translateY(0px); box-shadow: 0 15px 35px rgba(0,0,0,0.9); }
}

.navbar { display: flex; align-items: center; justify-content: space-between; padding: 25px 5%; background: linear-gradient(to bottom, rgba(0, 0, 0, 0.9) 0%, rgba(0, 0, 0, 0.4) 50%, rgba(0, 0, 0, 0) 100%); margin-bottom: -100px; position: relative; z-index: 50; animation: slideUpFade 0.8s ease-out; pointer-events: none; }

/* --- HIDDEN DEVELOPER EASTER EGG CSS --- */
.logo-container { position: relative; display: inline-flex; flex-direction: column; align-items: center; cursor: pointer; pointer-events: auto; }
.easter-egg { position: absolute; top: 15px; font-size: 13px; font-weight: 700; letter-spacing: 5px; color: transparent; transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); text-transform: uppercase; z-index: 1; }
.logo-container:hover .easter-egg { top: -18px; color: rgba(255, 255, 255, 0.8); text-shadow: 0 0 12px rgba(255,255,255,0.6); }
.logo { color: #E50914; font-size: 38px; font-weight: 900; letter-spacing: 2px; text-transform: uppercase; text-shadow: 0px 0px 20px rgba(229, 9, 20, 0.6); transition: all 0.3s ease; position: relative; z-index: 2; }
.logo-container:hover .logo { transform: scale(1.05); text-shadow: 0px 0px 25px rgba(229, 9, 20, 1); }
/* --------------------------------------- */

.hero-container { position: relative; width: 100%; height: 85vh; display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; overflow: hidden; background-color: #0b0b0c; border-bottom: 1px solid #1a1a1a; animation: slideUpFade 1s ease-out; }
.hero-bg-glow { position: absolute; top: -10%; left: -10%; width: 120%; height: 120%; background-size: cover; background-position: center; background-repeat: no-repeat; filter: blur(45px) brightness(0.35); z-index: 0; }
.hero-content { position: relative; z-index: 2; padding-left: 5%; width: 55%; margin-top: 40px; animation: slideUpFade 1.5s ease-out; }
.hero-title { font-size: 4.8rem; font-weight: 900; margin-bottom: 10px; line-height: 1.1; text-transform: uppercase; text-shadow: 3px 3px 8px rgba(0,0,0,0.9); letter-spacing: -1px; }
.hero-badge { background: linear-gradient(45deg, #E50914, #ff414d); color: white; padding: 6px 14px; border-radius: 4px; font-weight: 800; font-size: 0.9rem; margin-bottom: 25px; display: inline-block; box-shadow: 0 4px 10px rgba(229, 9, 20, 0.4); text-transform: uppercase; letter-spacing: 1px; }
.hero-desc { font-size: 1.25rem; color: #e5e5e5; text-shadow: 2px 2px 5px rgba(0,0,0,0.9); margin-bottom: 35px; font-weight: 400; line-height: 1.6; display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; max-width: 90%; }

.hero-buttons button { padding: 14px 30px; font-size: 1.2rem; font-weight: 700; border-radius: 6px; border: none; cursor: pointer; margin-right: 15px; transition: all 0.3s ease; display: inline-flex; align-items: center; gap: 8px;}
.btn-play { background-color: #ffffff; color: #000000; }
.btn-play:hover { background-color: #E50914; color: white; transform: scale(1.05); }
.btn-info { background-color: rgba(109, 109, 110, 0.6); color: white; backdrop-filter: blur(8px); }
.btn-info:hover { background-color: rgba(255, 255, 255, 0.25); transform: scale(1.05); }
.btn-hero-like { background-color: rgba(109, 109, 110, 0.6); color: white; backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.4) !important;}
.btn-hero-like:hover { background-color: rgba(255, 255, 255, 0.25); border-color: #E50914 !important; transform: scale(1.05); }
.btn-hero-like.liked { border-color: #E50914 !important; }

.hero-poster-box { position: relative; z-index: 2; width: 45%; display: flex; justify-content: center; align-items: center; padding-right: 5%; margin-top: 60px; animation: slideUpFade 1.2s ease-out; }
.hero-poster-box img { height: 55vh; border-radius: 12px; border: 1px solid rgba(255,255,255,0.15); animation: floatPoster 6s ease-in-out infinite;  }

.category-header { font-size: 1.6rem; color: #ffffff; font-weight: 700; margin-top: 40px; margin-bottom: 15px; padding-left: 4%; letter-spacing: 0.5px; animation: slideUpFade 1s ease-out both; }
.scroll-container { display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 20px; padding: 15px 4% 50px 4%; scroll-behavior: smooth; animation: slideUpFade 1.2s ease-out both; scroll-snap-type: x mandatory; -webkit-overflow-scrolling: touch; }
.scroll-container::-webkit-scrollbar { height: 8px; background: rgba(255,255,255,0.02); border-radius: 10px;} 
.scroll-container::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 10px;}
.scroll-container::-webkit-scrollbar-thumb:hover { background: #E50914; }

.like-btn { position: absolute; top: 10px; right: 10px; background-color: rgba(0, 0, 0, 0.6); color: white; border: 1px solid rgba(255,255,255,0.3); border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-size: 16px; text-decoration: none !important; z-index: 50; transition: all 0.2s ease; backdrop-filter: blur(4px); }
.like-btn:hover { transform: scale(1.15); background-color: rgba(0, 0, 0, 0.9); border-color: #E50914; }
.like-btn.liked { background-color: rgba(229, 9, 20, 0.15); border-color: #E50914; text-shadow: 0 0 10px rgba(229,9,20,0.8); }

.movie-card { flex: 0 0 240px; position: relative; transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); cursor: pointer; border-radius: 8px; scroll-snap-align: start; scroll-margin-left: 4%;}
.movie-card img { width: 100%; aspect-ratio: 2 / 3; object-fit: cover; border-radius: 8px; box-shadow: 0 6px 12px rgba(0,0,0,0.6); transition: all 0.4s ease; border: 2px solid transparent; }
.movie-card:hover { transform: scale(1.08) translateY(-10px); z-index: 10; }
.movie-card:hover img { border: 2px solid #E50914; box-shadow: 0 15px 30px rgba(229, 9, 20, 0.5); }

.top10-card { flex: 0 0 320px; display: flex; align-items: center; position: relative; padding-left: 30px; transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); cursor: pointer; scroll-snap-align: start; scroll-margin-left: 4%;}
.top10-number { font-size: 280px; font-weight: 900; color: #0b0b0c; -webkit-text-stroke: 4px #444; position: absolute; left: -20px; bottom: -50px; z-index: 1; letter-spacing: -15px; transition: all 0.5s ease; text-shadow: 5px 5px 10px rgba(0,0,0,0.8); }
.poster-wrapper { position: relative; z-index: 2; margin-left: 70px; transition: all 0.4s ease; display: flex; }
.top10-card img { width: 200px; aspect-ratio: 2 / 3; object-fit: cover; border-radius: 8px; box-shadow: 0 8px 16px rgba(0,0,0,0.8); border: 2px solid transparent; transition: all 0.4s ease;}
.top10-card:hover { transform: translateY(-10px); z-index: 10; }
.top10-card:hover .poster-wrapper { transform: scale(1.1) rotate(2deg); }
.top10-card:hover img { border: 2px solid #E50914; box-shadow: 0 15px 35px rgba(229, 9, 20, 0.6); }
.top10-card:hover .top10-number { color: rgba(229,9,20,0.1); -webkit-text-stroke: 4px #E50914; transform: scale(1.05) translateX(-10px); text-shadow: 0 0 20px rgba(229,9,20,0.4); }

.stTextInput { position: absolute; top: 15px; right: 5%; width: 280px !important; z-index: 100 !important; pointer-events: auto;}
.stTextInput input { color: white !important; background-color: rgba(0, 0, 0, 0.6) !important; border: 1px solid rgba(255, 255, 255, 0.2) !important; border-radius: 6px !important; padding: 12px 20px !important; font-size: 15px !important; transition: all 0.3s ease !important; }
.stTextInput input::placeholder { color: rgba(255,255,255,0.4) !important; font-weight: 300 !important; }
.stTextInput input:focus { background-color: rgba(0, 0, 0, 0.9) !important; border-color: #E50914 !important; box-shadow: none !important; width: 320px !important; }
</style>
""", unsafe_allow_html=True)

# --- MAGIC JAVASCRIPT INJECTION ---
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

# --- NAVIGATION BAR WITH HIDDEN EASTER EGG ---
st.markdown("""
<div class="navbar">
    <div class="logo-container">
        <div class="easter-egg">bclol87</div>
        <div class="logo">TARFLIX</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- SEARCH BAR WITH STATE TRACKING ---
search_query = st.text_input("", placeholder="🔍 Search titles, genres, actors...", label_visibility="collapsed", key="search_query")

if search_query:
    st.query_params["q"] = search_query
elif "q" in st.query_params:
    del st.query_params["q"]

# --- HELPER FUNCTION TO RENDER UI CARDS ---
def render_movie_cards(recommendations, score_column, is_top_10=False):
    container_class = "scroll-container top10-scroll-row" if is_top_10 else "scroll-container"
    html_content = f'<div class="{container_class}">'
    
    current_q = st.session_state.get("search_query", "")
    q_param = f"&q={urllib.parse.quote(current_q)}" if current_q else ""
    
    for i, (_, row) in enumerate(recommendations.iterrows()):
        title = row['title']
        poster_url, overview, movie_link = fetch_movie_details(title)
        score = row.get(score_column, 85) 
        title_safe = str(title).replace('"', '&quot;')
        
        is_liked = title in st.session_state.liked_movies
        future_likes = st.session_state.liked_movies.copy()
        if is_liked:
            future_likes.remove(title)
        else:
            future_likes.append(title)
            
        likes_str = "|".join(future_likes)
        heart_icon = "❤️" if is_liked else "🤍"
        btn_class = "like-btn liked" if is_liked else "like-btn"
        like_url = f"/?likes={urllib.parse.quote(likes_str)}{q_param}"
        like_html = f'<a href="{like_url}" target="_self" class="{btn_class}" title="Like {title_safe}">{heart_icon}</a>'
        
        if is_top_10:
            rank = i + 1
            html_content += f'<div class="top10-card"><div class="top10-number">{rank}</div><div class="poster-wrapper">{like_html}<a href="{movie_link}" target="_blank"><img src="{poster_url}" alt="{title_safe}"></a></div></div>'
        else:
            html_content += f'<div class="movie-card" title="{title_safe} - {score:.0f}% Match">{like_html}<a href="{movie_link}" target="_blank"><img src="{poster_url}" alt="{title_safe}"></a></div>'
            
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

# --- RESULTS SECTION ---
if search_query:
    with st.spinner('Curating cinematic experience...'):
        
        # --- 1. SMART TEXT CLEANER ---
        clean_query = search_query.lower()
        fillers = [r'\bmovies\b', r'\bmovie\b', r'\brelated\b', r'\babout\b', r'\bshow\b', r'\bme\b', r'\bfind\b', r'\bi\b', r'\bwant\b', r'\bto\b', r'\bwatch\b', r'\bsome\b', r'\ba\b', r'\bthe\b']
        for filler in fillers:
            clean_query = re.sub(filler, '', clean_query).strip()
            
        words = clean_query.split()
        clean_query = " ".join(sorted(set(words), key=words.index))
        
        if not clean_query:
            clean_query = search_query

        # --- 2. THE AI DECISION MAKER ---
        exact_title = movies[movies['title'].str.lower() == clean_query]
        
        query_vec = tfidf.transform([clean_query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_match_idx = sim_scores.argmax()
        best_score = sim_scores[best_match_idx]
        
        if not exact_title.empty or best_score > 0.35:
            selected_movie = exact_title.iloc[0]['title'] if not exact_title.empty else movies.iloc[best_match_idx]['title']
            hero_poster, hero_overview, hero_link = fetch_movie_details(selected_movie)
            
            is_hero_liked = selected_movie in st.session_state.liked_movies
            hero_future_likes = st.session_state.liked_movies.copy()
            if is_hero_liked:
                hero_future_likes.remove(selected_movie)
            else:
                hero_future_likes.append(selected_movie)
                
            hero_likes_str = "|".join(hero_future_likes)
            current_q = st.session_state.get("search_query", "")
            q_param = f"&q={urllib.parse.quote(current_q)}" if current_q else ""
            hero_like_url = f"/?likes={urllib.parse.quote(hero_likes_str)}{q_param}"
            
            hero_heart = "❤️" if is_hero_liked else "🤍"
            hero_like_text = "Liked" if is_hero_liked else "Like"
            hero_btn_class = "btn-hero-like liked" if is_hero_liked else "btn-hero-like"
            
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

<a href="{hero_like_url}" target="_self" style="text-decoration:none;">
<button class="btn-hero-like {hero_btn_class}">{hero_heart} {hero_like_text}</button>
</a>

</div>
</div>
<div class="hero-poster-box">
<img src="{hero_poster}" alt="{selected_movie}">
</div>
</div>
""", unsafe_allow_html=True)

            if st.session_state.liked_movies:
                st.markdown('<div class="category-header">🎬 Your Liked Collection</div>', unsafe_allow_html=True)
                liked_df = movies[movies['title'].isin(st.session_state.liked_movies)].copy()
                if not liked_df.empty:
                    liked_df['My_Score'] = 100 
                    render_movie_cards(liked_df, 'My_Score')

                st.markdown('<div class="category-header">❤️ Because of your Liked Movies</div>', unsafe_allow_html=True)
                profile_recs = get_profile_based_recs(st.session_state.liked_movies)
                if not profile_recs.empty:
                    render_movie_cards(profile_recs, 'Profile_Score')
            
            st.markdown('<div class="category-header">Top 10 Movies in Your Area Today</div>', unsafe_allow_html=True)
            render_movie_cards(get_community_recs(selected_movie).head(10), 'CF_Score', is_top_10=True)

            st.markdown(f'<div class="category-header">Because you searched for {selected_movie}</div>', unsafe_allow_html=True)
            render_movie_cards(get_content_based_recs(selected_movie), 'CB_Score')

            st.markdown('<div class="category-header">AI Predictions: You\'ll Love These</div>', unsafe_allow_html=True)
            render_movie_cards(get_hybrid_recs(selected_movie), 'Hybrid_Score')

        else:
            st.markdown(f"""
            <div class="hero-container" style="height: 35vh; min-height: 250px;">
                <div class="hero-bg-glow" style="background-image: url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=2070'); opacity: 0.3;"></div>
                <div class="hero-content" style="width: 100%; text-align: center; margin-top: 0; padding: 0;">
                    <div class="hero-title" style="font-size: 3.5rem;">Results for "{clean_query.title()}"</div>
                    <div class="hero-desc" style="max-width: 100%;">Exploring the global database for your topic...</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            topic_results = search_tmdb_topic(clean_query)
            
            if topic_results:
                st.markdown('<div class="category-header">Global Topic Matches</div>', unsafe_allow_html=True)
                topic_df = pd.DataFrame(topic_results)
                render_movie_cards(topic_df, 'score')
                
                if best_score > 0.05:
                    st.markdown('<div class="category-header">From our Local Library</div>', unsafe_allow_html=True)
                    top_indices = sim_scores.argsort()[::-1][:15]
                    local_recs = movies.iloc[top_indices].copy()
                    local_recs['score'] = sim_scores[top_indices] * 100
                    render_movie_cards(local_recs, 'score')
            else:
                st.error("No movies found for that search. Try another keyword!")

else:
    # --- HOME SCREEN ---
    st.markdown("""
<div class="hero-container">
<div class="hero-bg-glow" style="background-image: url('https://assets.nflxext.com/ffe/siteui/vlv3/1ecf18b2-adad-4684-bd9a-acab7f2a875f/728df0cc-b789-4bba-9ea7-626a5c2d36ab/MY-en-20230116-popsignuptwoweeks-perspective_alpha_website_medium.jpg'); opacity: 0.5;"></div>
<div class="hero-content" style="width: 100%; text-align: center; padding: 0; margin-top: 0;">
<div class="hero-title" style="font-size: 3.5rem; text-shadow: 2px 4px 10px rgba(0,0,0,0.8);">FIND YOUR NEXT OBSESSION</div>
<div class="hero-desc" style="color:#ddd; font-size: 1.4rem;">Type a movie title or mood in the top right to unleash the recommendation engine.</div>
</div>
</div>
""", unsafe_allow_html=True)

    if st.session_state.liked_movies:
        st.markdown('<div class="category-header">🎬 Your Liked Collection</div>', unsafe_allow_html=True)
        liked_df = movies[movies['title'].isin(st.session_state.liked_movies)].copy()
        if not liked_df.empty:
            liked_df['My_Score'] = 100
            render_movie_cards(liked_df, 'My_Score')

        st.markdown('<div class="category-header">❤️ For You (Based on your Likes)</div>', unsafe_allow_html=True)
        profile_recs = get_profile_based_recs(st.session_state.liked_movies)
        if not profile_recs.empty:
            render_movie_cards(profile_recs, 'Profile_Score')
