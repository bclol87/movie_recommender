import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- Import logic from movie_logic.py ---
# Note: Ensure this module is in the same directory and contains the required functions/variables.
from movie_logic import (
    movies, cosine_sim, tfidf, tfidf_matrix,
    fetch_movie_details, search_tmdb_topic,
    get_content_based_recs, get_community_recs, get_hybrid_recs
)

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="Zmovo (Netflix Style)", page_icon="🍿", layout="wide")

# Netflix Theme CSS
st.markdown("""
    <style>
    /* Main Background & Text */
    .stApp { background-color: #141414 !important; color: #e5e5e5 !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    
    /* Hide default Streamlit headers/footers/padding */
    .block-container { padding-top: 0rem !important; padding-left: 0 !important; padding-right: 0 !important; max-width: 100% !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

    /* Top Navigation */
    .top-nav { display: flex; justify-content: space-between; align-items: center; padding: 20px 4%; background: linear-gradient(to bottom, rgba(0,0,0,0.8) 0%, rgba(20,20,20,0) 100%); position: relative; z-index: 10;}
    .nav-left { display: flex; align-items: center; gap: 40px; }
    .nav-logo { font-size: 1.8rem; font-weight: 900; color: #E50914; letter-spacing: 1px; }
    .nav-links { display: flex; gap: 20px; font-size: 0.85rem; font-weight: 500; }
    .nav-links span { color: #e5e5e5; cursor: pointer; transition: color 0.4s; }
    .nav-links span.active { color: #ffffff; font-weight: 700; }
    .nav-links span:hover { color: #b3b3b3; }
    .nav-right { display: flex; align-items: center; gap: 20px; font-size: 1.2rem; cursor: pointer; }

    /* Sub Navigation (Movies + Genres) */
    .sub-nav { display: flex; align-items: center; gap: 20px; padding: 0 4%; margin-top: -10px; position: relative; z-index: 10;}
    .sub-nav h2 { font-size: 2rem; color: white; margin: 0; font-weight: 700; }
    .genre-select { background-color: #000; border: 1px solid rgba(255,255,255,0.2); color: white; padding: 5px 10px; font-weight: 600; font-size: 0.85rem; cursor: pointer; }

    /* Streamlit Search Bar Override */
    .search-container { padding: 0 4%; margin-top: 20px; position: relative; z-index: 10; max-width: 400px; }
    div[data-testid="stTextInput"] input { background-color: rgba(0,0,0,0.75) !important; color: white !important; border: 1px solid #e5e5e5 !important; border-radius: 0px !important; padding: 10px 15px !important;}
    div[data-testid="stTextInput"] input:focus { border-color: white !important; box-shadow: none !important;}

    /* Hero Banner */
    .hero-banner { 
        background-size: cover; background-position: center top; 
        height: 80vh; margin-top: -120px; 
        display: flex; flex-direction: column; justify-content: center; padding: 0 4%;
        position: relative;
    }
    .hero-vignette { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(77deg, rgba(0,0,0,.6) 0, rgba(0,0,0,0) 85%); pointer-events: none;}
    .hero-bottom-fade { position: absolute; bottom: 0; left: 0; right: 0; height: 150px; background: linear-gradient(to top, #141414 0%, rgba(20,20,20,0) 100%); pointer-events: none;}
    .hero-content { position: relative; z-index: 5; max-width: 40%; margin-top: 100px;}
    
    .hero-title { font-size: 4.5rem; font-weight: 900; margin: 0 0 10px 0; color: #fff; line-height: 1.1; text-shadow: 2px 2px 4px rgba(0,0,0,0.45);}
    .hero-top10 { display: flex; align-items: center; gap: 10px; font-weight: bold; font-size: 1.2rem; margin-bottom: 15px; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);}
    .top10-badge { background-color: #E50914; color: white; padding: 2px 6px; font-size: 0.7rem; font-weight: 900; border-radius: 2px;}
    
    .hero-meta { color: #fff; font-size: 1.2rem; margin-bottom: 25px; line-height: 1.4; font-weight: 400; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;}
    
    /* Netflix Buttons */
    .btn-row { display: flex; gap: 15px; }
    .btn-play { background-color: white; color: black; padding: 8px 24px; border-radius: 4px; font-weight: 700; font-size: 1.2rem; text-decoration: none; display: flex; align-items: center; gap: 10px; transition: 0.2s;}
    .btn-play:hover { background-color: rgba(255,255,255,0.7); }
    .btn-info { background-color: rgba(109, 109, 110, 0.7); color: white; padding: 8px 24px; border-radius: 4px; font-weight: 700; font-size: 1.2rem; text-decoration: none; display: flex; align-items: center; gap: 10px; transition: 0.2s;}
    .btn-info:hover { background-color: rgba(109, 109, 110, 0.4); }

    /* Section Headers */
    .section-container { padding: 0 4%; margin-top: -30px; position: relative; z-index: 10;}
    .section-title { font-size: 1.2rem; font-weight: 700; color: #e5e5e5; margin-bottom: 10px; }

    /* Netflix Scroll Container & Horizontal Cards */
    .scroll-container { display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 10px; padding-bottom: 40px; scroll-behavior: smooth; }
    .scroll-container::-webkit-scrollbar { display: none; } /* Hide scrollbar for clean look */
    
    .z-card { flex: 0 0 280px; aspect-ratio: 16/9; background-color: #222; border-radius: 4px; overflow: hidden; position: relative; cursor: pointer; transition: transform 0.3s ease, z-index 0.3s ease; }
    .z-card:hover { transform: scale(1.08); z-index: 20; box-shadow: 0 10px 20px rgba(0,0,0,0.8); border-radius: 4px;}
    
    .z-card img { width: 100%; height: 100%; object-fit: cover; object-position: center 20%; }
    
    /* Card Badges / Hover info */
    .card-top10-badge { position: absolute; top: 5px; right: 5px; background-color: #E50914; color: white; font-size: 0.6rem; font-weight: 900; padding: 2px 4px; border-radius: 2px;}
    .card-recently-added { position: absolute; bottom: 0; left: 0; right: 0; background-color: #E50914; color: white; font-size: 0.7rem; font-weight: bold; text-align: center; padding: 4px 0;}
    
    .card-hover-info { position: absolute; bottom: 0; left: 0; right: 0; background: linear-gradient(to top, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0) 100%); padding: 10px; opacity: 0; transition: opacity 0.3s;}
    .z-card:hover .card-hover-info { opacity: 1; }
    .card-hover-title { font-size: 0.9rem; font-weight: bold; color: white; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .card-hover-match { color: #46d369; font-weight: bold; font-size: 0.8rem;}

    </style>
""", unsafe_allow_html=True)

# --- TOP NAVIGATION BAR ---
st.markdown("""
    <div class="top-nav">
        <div class="nav-left">
            <div class="nav-logo">ZMOVO</div>
            <div class="nav-links">
                <span>Home</span>
                <span>Shows</span>
                <span class="active">Movies</span>
                <span>Games</span>
                <span>New & Popular</span>
                <span>My List</span>
                <span>Browse by Languages</span>
            </div>
        </div>
        <div class="nav-right">
            <span>🔍</span> <span>🔔</span> <span style="background: #333; padding: 4px; border-radius: 4px;">🟨</span>
        </div>
    </div>
    
    <div class="sub-nav">
        <h2>Movies</h2>
        <select class="genre-select">
            <option>Genres</option>
            <option>Action</option>
            <option>Comedy</option>
            <option>Drama</option>
        </select>
    </div>
""", unsafe_allow_html=True)


# --- HELPER FUNCTION TO RENDER NETFLIX-STYLE CARDS ---
def render_movie_cards(recommendations, score_column, show_top10=False, show_recent=False):
    html_content = '<div class="scroll-container">'
    
    for i, (_, row) in enumerate(recommendations.iterrows()):
        poster_url, overview, movie_link = fetch_movie_details(row['title'])
        score = row.get(score_column, 85)
        
        # Badges toggles
        top10_html = '<div class="card-top10-badge">TOP<br>10</div>' if show_top10 else ''
        recent_html = '<div class="card-recently-added">Recently Added</div>' if show_recent else ''

        html_content += f"""<div class="z-card" onclick="window.open('{movie_link}', '_blank')">
            <img src="{poster_url}" alt="{row['title']}">
            {top10_html}
            {recent_html}
            <div class="card-hover-info">
                <div class="card-hover-match">{score:.0f}% Match</div>
                <div class="card-hover-title">{row['title']}</div>
            </div>
        </div>"""
        
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)


# --- SEARCH INPUT ---
st.markdown('<div class="search-container">', unsafe_allow_html=True)
search_query = st.text_input("Search", placeholder="Search titles, people, genres...", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)


# --- MAIN LOGIC & RENDERING ---
if search_query:
    with st.spinner('Loading...'):
        query_vec = tfidf.transform([search_query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_match_idx = sim_scores.argmax()
        best_score = sim_scores[best_match_idx]
        
        if best_score > 0:
            selected_movie = movies.iloc[best_match_idx]['title']
            poster_url, overview, _ = fetch_movie_details(selected_movie)
            
            # HERO BANNER
            hero_html = f"""
            <div class="hero-banner" style="background-image: url('{poster_url}');">
                <div class="hero-vignette"></div>
                <div class="hero-bottom-fade"></div>
                <div class="hero-content">
                    <h1 class="hero-title" style="text-transform: uppercase;">{selected_movie}</h1>
                    <div class="hero-top10">
                        <span class="top10-badge">TOP<br>10</span> #1 in Movies Today
                    </div>
                    <p class="hero-meta">{overview if overview else 'Desperate to survive a night of violence, a newly freed ex-con must protect police from a ruthless gang.'}</p>
                    <div class="btn-row">
                        <a href="#" class="btn-play">▶ Play</a>
                        <a href="#" class="btn-info">ⓘ More Info</a>
                    </div>
                </div>
            </div>
            """
            st.markdown(hero_html, unsafe_allow_html=True)
            
            # SECTIONS (Full width)
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            
            st.markdown('<div class="section-title">New on Zmovo</div>', unsafe_allow_html=True)
            render_movie_cards(get_hybrid_recs(selected_movie), 'Hybrid_Score', show_top10=True)
            
            st.markdown('<div class="section-title">We Think You\'ll Love These</div>', unsafe_allow_html=True)
            render_movie_cards(get_community_recs(selected_movie), 'CF_Score')
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.markdown('<div class="section-container"><div class="section-title">More to explore</div>', unsafe_allow_html=True)
            topic_results = search_tmdb_topic(search_query)
            if topic_results:
                render_movie_cards(pd.DataFrame(topic_results), 'score')
            else:
                st.warning("No matches found.")
            st.markdown('</div>', unsafe_allow_html=True)

else:
    # Default State (No Search)
    st.markdown("""
        <div class="hero-banner" style="background-image: url('https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80');">
            <div class="hero-vignette"></div>
            <div class="hero-bottom-fade"></div>
            <div class="hero-content">
                <h1 class="hero-title">WELCOME TO ZMOVO</h1>
                <p class="hero-meta">Discover blockbuster movies, epic TV shows, and award-winning originals. Search above to start browsing.</p>
                <div class="btn-row">
                    <a href="#" class="btn-play">▶ Play</a>
                    <a href="#" class="btn-info">ⓘ More Info</a>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top 10 Movies in Malaysia Today</div>', unsafe_allow_html=True)
    # Renders the popular movies with a "Recently Added" badge and "Top 10" badge to mirror the screenshot
    render_movie_cards(movies.sort_values('vote_count', ascending=False).head(10), 'vote_average', show_recent=True)
    st.markdown('</div>', unsafe_allow_html=True)
