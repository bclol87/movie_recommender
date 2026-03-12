import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- Import logic from movie_logic.py ---
# Make sure your movie_logic.py is in the same folder!
from movie_logic import (
    movies, cosine_sim, tfidf, tfidf_matrix,
    fetch_movie_details, search_tmdb_topic,
    get_content_based_recs, get_community_recs, get_hybrid_recs
)

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="Zmovo Movies", page_icon="🍿", layout="wide")

st.markdown("""
    <style>
    /* Main Background & Text */
    .stApp { background-color: #141414 !important; color: #e5e5e5 !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; overflow-x: hidden; }
    
    /* Hide default Streamlit headers/footers/padding */
    .block-container { padding-top: 0rem !important; padding-left: 0 !important; padding-right: 0 !important; max-width: 100% !important; overflow: hidden; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

    /* Top Navigation Header Mockup */
    .top-header { 
        position: absolute; top: 0; left: 0; right: 0; height: 70px; 
        background: linear-gradient(to bottom, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0) 100%); 
        z-index: 100; display: flex; align-items: center; justify-content: space-between; padding: 0 4%;
    }
    .header-logo { color: #E50914; font-size: 1.8rem; font-weight: 900; letter-spacing: 1px; }
    .header-links { display: flex; gap: 20px; font-size: 0.9rem; font-weight: 500; margin-left: 40px; flex-grow: 1;}
    .header-links span { color: #e5e5e5; cursor: pointer; transition: 0.3s; }
    .header-links span.active { color: white; font-weight: bold; }
    .header-links span:hover { color: #b3b3b3; }

    /* Streamlit Search Bar Override (Floating in Header) */
    div[data-testid="stTextInput"] { position: absolute; top: 15px; right: 8%; z-index: 1000; width: 300px; }
    div[data-testid="stTextInput"] input { background-color: rgba(0,0,0,0.7) !important; color: white !important; border: 1px solid rgba(255,255,255,0.4) !important; border-radius: 4px !important; padding: 8px 15px !important; font-size: 0.9rem !important;}
    div[data-testid="stTextInput"] input:focus { border-color: white !important; box-shadow: none !important; background-color: rgba(0,0,0,0.9) !important;}

    /* Hero Banner */
    .hero-banner { 
        background-size: cover; background-position: center top; 
        height: 85vh; 
        display: flex; flex-direction: column; justify-content: center; padding: 0 4%;
        position: relative;
    }
    .hero-vignette { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(77deg, rgba(0,0,0,.8) 0, rgba(0,0,0,0) 85%); pointer-events: none;}
    .hero-bottom-fade { position: absolute; bottom: 0; left: 0; right: 0; height: 150px; background: linear-gradient(to top, #141414 0%, rgba(20,20,20,0) 100%); pointer-events: none;}
    .hero-content { position: relative; z-index: 5; max-width: 45%; margin-top: 60px;}
    
    .hero-title { font-size: 5rem; font-weight: 900; margin: 0 0 10px 0; color: #fff; line-height: 1.1; text-transform: uppercase; text-shadow: 2px 2px 4px rgba(0,0,0,0.45); font-family: 'Impact', sans-serif; letter-spacing: 2px;}
    .hero-top10 { display: flex; align-items: center; gap: 10px; font-weight: bold; font-size: 1.2rem; margin-bottom: 15px; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);}
    .top10-badge { background-color: #E50914; color: white; padding: 2px 6px; font-size: 0.8rem; font-weight: 900; border-radius: 2px; text-align: center; line-height: 1.1;}
    
    .hero-meta { color: #fff; font-size: 1.2rem; margin-bottom: 25px; line-height: 1.4; font-weight: 400; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;}
    
    /* Buttons */
    .btn-row { display: flex; gap: 15px; }
    .btn-play { background-color: white; color: black; padding: 10px 28px; border-radius: 4px; font-weight: 700; font-size: 1.2rem; text-decoration: none; display: flex; align-items: center; gap: 10px; transition: 0.2s;}
    .btn-play:hover { background-color: rgba(255,255,255,0.7); }
    .btn-info { background-color: rgba(109, 109, 110, 0.7); color: white; padding: 10px 28px; border-radius: 4px; font-weight: 700; font-size: 1.2rem; text-decoration: none; display: flex; align-items: center; gap: 10px; transition: 0.2s;}
    .btn-info:hover { background-color: rgba(109, 109, 110, 0.4); }

    /* Section Headers */
    .section-container { padding: 0 4%; margin-top: -30px; position: relative; z-index: 10;}
    .section-title { font-size: 1.3rem; font-weight: 700; color: #e5e5e5; margin-bottom: 15px; margin-top: 30px;}

    /* Horizontal Scrolling Containers */
    .scroll-container { display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 12px; padding-bottom: 20px; scroll-behavior: smooth; }
    .scroll-container::-webkit-scrollbar { display: none; } 
    
    /* Standard 16:9 Card */
    .z-card { flex: 0 0 280px; aspect-ratio: 16/9; background-color: #222; border-radius: 4px; overflow: hidden; position: relative; cursor: pointer; transition: transform 0.3s ease, z-index 0.3s ease; border: 1px solid rgba(255,255,255,0.05);}
    .z-card:hover { transform: scale(1.08); z-index: 20; box-shadow: 0 10px 20px rgba(0,0,0,0.8); border-radius: 4px;}
    .z-card img { width: 100%; height: 100%; object-fit: cover; object-position: center 20%; }
    
    /* Top 10 Giant Number Wrapper */
    .top10-wrapper { flex: 0 0 240px; display: flex; position: relative; cursor: pointer; margin-right: 45px; transition: transform 0.3s ease;}
    .top10-wrapper:hover { transform: scale(1.05); z-index: 20; }
    .top10-number { font-size: 16rem; font-weight: 900; color: #000; -webkit-text-stroke: 4px #555; position: absolute; left: -35px; bottom: -65px; z-index: 1; letter-spacing: -10px; line-height: 1;}
    .top10-card { flex: 1; aspect-ratio: 2/3; background-color: #222; border-radius: 4px; overflow: hidden; position: relative; z-index: 2; margin-left: 60px; border: 1px solid rgba(255,255,255,0.1); box-shadow: 5px 0 15px rgba(0,0,0,0.6);}
    .top10-card img { width: 100%; height: 100%; object-fit: cover; }

    /* Badges */
    .card-top10-badge { position: absolute; top: 5px; right: 5px; background-color: #E50914; color: white; font-size: 0.6rem; font-weight: 900; padding: 2px 4px; border-radius: 2px; text-align: center; line-height: 1.1;}
    .card-recently-added { position: absolute; bottom: 0; left: 0; right: 0; background-color: #E50914; color: white; font-size: 0.75rem; font-weight: bold; text-align: center; padding: 5px 0;}
    
    /* Hover Info */
    .card-hover-info { position: absolute; bottom: 0; left: 0; right: 0; background: linear-gradient(to top, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0) 100%); padding: 10px; opacity: 0; transition: opacity 0.3s;}
    .z-card:hover .card-hover-info { opacity: 1; }
    .card-hover-title { font-size: 0.9rem; font-weight: bold; color: white; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .card-hover-match { color: #46d369; font-weight: bold; font-size: 0.8rem;}
    </style>
""", unsafe_allow_html=True)

# --- HEADER NAVIGATION (HTML) ---
st.markdown("""
    <div class="top-header">
        <div class="header-logo">ZMOVO</div>
        <div class="header-links">
            <span>Home</span>
            <span>Shows</span>
            <span class="active">Movies</span>
            <span>New & Popular</span>
            <span>My List</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- SEARCH INPUT (Floats in top right via CSS) ---
search_query = st.text_input("Search", placeholder="🔍 Titles, people, genres...", label_visibility="collapsed")

# --- HELPER FUNCTION TO RENDER CARDS ---
def render_movie_cards(recommendations, score_column, is_top10_row=False, show_recent=False, show_top10_badge=False):
    html_content = '<div class="scroll-container">'
    
    for i, (_, row) in enumerate(recommendations.iterrows()):
        poster_url, overview, movie_link = fetch_movie_details(row['title'])
        score = row.get(score_column, 85)
        
        recent_html = '<div class="card-recently-added">Recently Added</div>' if show_recent else ''
        badge_html = '<div class="card-top10-badge">TOP<br>10</div>' if show_top10_badge else ''

        if is_top10_row:
            html_content += f"""
            <div class="top10-wrapper" onclick="window.open('{movie_link}', '_blank')">
                <div class="top10-number">{i+1}</div>
                <div class="top10-card">
                    <img src="{poster_url}" alt="{row['title']}">
                    {recent_html}
                </div>
            </div>"""
        else:
            html_content += f"""
            <div class="z-card" onclick="window.open('{movie_link}', '_blank')">
                <img src="{poster_url}" alt="{row['title']}">
                {badge_html}
                {recent_html}
                <div class="card-hover-info">
                    <div class="card-hover-match">{score:.0f}% Match</div>
                    <div class="card-hover-title">{row['title']}</div>
                </div>
            </div>"""
        
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

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
            
            # HERO BANNER FOR SEARCH RESULTS
            hero_html = f"""
            <div class="hero-banner" style="background-image: url('{poster_url}');">
                <div class="hero-vignette"></div>
                <div class="hero-bottom-fade"></div>
                <div class="hero-content">
                    <h1 class="hero-title">{selected_movie}</h1>
                    <div class="hero-top10">
                        <span class="top10-badge">TOP<br>10</span> #1 in Search Results
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
            
            # SECTIONS
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            
            st.markdown('<div class="section-title">Similar to your search</div>', unsafe_allow_html=True)
            render_movie_cards(get_hybrid_recs(selected_movie), 'Hybrid_Score', show_top10_badge=True)
            
            st.markdown('<div class="section-title">We Think You\'ll Love These</div>', unsafe_allow_html=True)
            render_movie_cards(get_community_recs(selected_movie), 'CF_Score')
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.markdown('<br><br><br><div class="section-container"><div class="section-title">Global Search Results</div>', unsafe_allow_html=True)
            topic_results = search_tmdb_topic(search_query)
            if topic_results:
                render_movie_cards(pd.DataFrame(topic_results), 'score')
            else:
                st.warning("No movies found.")
            st.markdown('</div>', unsafe_allow_html=True)

else:
    # DEFAULT STATE (Matches the mockup exactly)
    st.markdown("""
        <div class="hero-banner" style="background-image: url('https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80');">
            <div class="hero-vignette"></div>
            <div class="hero-bottom-fade"></div>
            <div class="hero-content">
                <h1 class="hero-title">BANDUAN</h1>
                <div class="hero-top10">
                    <span class="top10-badge">TOP<br>10</span> #1 in Movies Today
                </div>
                <p class="hero-meta">Desperate to meet his young daughter, a newly freed ex-con must survive a night of violence after he's forced to protect police from a ruthless gang.</p>
                <div class="btn-row">
                    <a href="#" class="btn-play">▶ Play</a>
                    <a href="#" class="btn-info">ⓘ More Info</a>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    
    # ROW 1: Giant Numbers Row
    st.markdown('<div class="section-title">Top 10 Movies in Malaysia Today</div>', unsafe_allow_html=True)
    render_movie_cards(movies.sort_values('vote_count', ascending=False).head(10), 'vote_average', is_top10_row=True, show_recent=True)
    
    # ROW 2: Standard Horizontal Cards
    st.markdown('<div class="section-title">New on Zmovo</div>', unsafe_allow_html=True)
    render_movie_cards(movies.head(10), 'vote_average', show_top10_badge=True)

    # ROW 3: Standard Horizontal Cards
    st.markdown('<div class="section-title">We Think You\'ll Love These</div>', unsafe_allow_html=True)
    render_movie_cards(movies.sample(frac=1).head(10), 'vote_average')
    
    st.markdown('</div>', unsafe_allow_html=True)
