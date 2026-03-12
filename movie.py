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
# Added initial_sidebar_state to mimic the Prime Video side menu
st.set_page_config(page_title="Prime CineMatch", page_icon="🍿", layout="wide", initial_sidebar_state="expanded")

# --- CINEMATIC UI (PRIME VIDEO INSPIRED) CSS ---
st.markdown("""
    <style>
    /* Global Dark Theme */
    .stApp { background-color: #0f171e !important; color: #ffffff !important; }
    
    /* Hide top padding */
    .block-container { padding-top: 2rem !important; }

    /* Sidebar Styling to match Prime Video */
    [data-testid="stSidebar"] { background-color: #0f171e !important; border-right: 1px solid #1f2b36; }
    
    /* Top Bar Inputs */
    .stTextInput input {
        background-color: #f2f4f8 !important; /* Light input bar */
        color: #000000 !important;
        border-radius: 4px;
        border: none !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1a242f !important;
        color: white !important;
        border: 1px solid #303b44 !important;
    }

    /* Section Headers */
    .category-header { 
        font-size: 1.5rem; 
        color: #ffffff !important; 
        font-weight: 600; 
        text-align: center;
        margin-top: 1rem; 
        margin-bottom: 1rem; 
    }

    /* Horizontal Scroll Container */
    .scroll-container { 
        display: flex; 
        flex-wrap: nowrap; 
        overflow-x: auto; 
        gap: 20px; 
        padding: 30px 10px; 
        scroll-behavior: smooth;
        align-items: flex-end; /* Aligns bottoms of cards */
    }
    .scroll-container::-webkit-scrollbar { display: none; }

    /* Prime Card Base */
    .movie-card { 
        flex: 0 0 220px; 
        background-color: transparent; 
        border-radius: 12px; 
        transition: transform 0.3s ease-out, z-index 0.3s;
        cursor: pointer;
        display: flex;
        flex-direction: column;
    }
    
    /* Hover Effect: Scale Up */
    .movie-card:hover { 
        transform: scale(1.15); 
        z-index: 100;
    }

    .movie-poster { 
        width: 100%; 
        aspect-ratio: 2/3;
        object-fit: cover;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        transition: border-radius 0.3s;
    }

    /* Info overlay background changes on hover */
    .card-overlay {
        padding: 15px 10px 10px 10px;
        opacity: 0.9;
        transition: all 0.3s;
        background-color: transparent;
    }
    
    .movie-card:hover .card-overlay {
        background-color: #1a242f;
        box-shadow: 0 20px 40px rgba(0,0,0,0.8);
        border: 1px solid #303b44;
        border-top: none;
        border-bottom-left-radius: 12px;
        border-bottom-right-radius: 12px;
    }
    
    .movie-card:hover .movie-poster {
        border-bottom-left-radius: 0px;
        border-bottom-right-radius: 0px;
    }

    .movie-title { 
        font-size: 1.05rem; 
        color: #ffffff !important; 
        font-weight: 600; 
        margin-bottom: 5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .match-score { 
        color: #46d369 !important; /* Prime Green */
        font-weight: 600; 
        font-size: 0.85rem; 
        margin-bottom: 12px;
    }
    .reason-text { color: #8197a4; font-weight: 400; }

    .watch-btn { 
        background-color: #00A8E1; 
        color: #ffffff !important; 
        padding: 8px; 
        border-radius: 4px; 
        text-decoration: none; 
        font-weight: 700; 
        display: block; 
        text-align: center;
        font-size: 0.85rem;
        transition: background-color 0.2s;
        opacity: 0; /* Hidden until hover */
    }
    .movie-card:hover .watch-btn { opacity: 1; }
    .watch-btn:hover { background-color: #008ebf; }

    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAV ---
with st.sidebar:
    st.markdown("<h2 style='color:white; margin-bottom: 2rem;'>Search</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:white; font-weight:400; margin-bottom: 1rem;'>Home</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:white; font-weight:400; margin-bottom: 1rem;'>Tv shows</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:white; font-weight:400; margin-bottom: 1rem;'>Movies</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:white; font-weight:400; margin-bottom: 1rem;'>Newest</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:white; font-weight:400; margin-bottom: 1rem;'>My list</h4>", unsafe_allow_html=True)

# --- TOP SEARCH & FILTER BAR ---
col1, col2, col3, col4 = st.columns([2, 4, 1, 2])
with col1:
    search_query = st.text_input("Search", placeholder="Search titles, actors...", label_visibility="collapsed")
with col2:
    # Spacer for the center logo area
    st.markdown("<h3 style='text-align: center; color: white; margin-top: 0;'>prime video</h3>", unsafe_allow_html=True)
with col3:
    st.markdown("<div style='text-align: right; color: white; padding-top: 8px; font-size: 0.9rem;'>FILTER:</div>", unsafe_allow_html=True)
with col4:
    selected_display = st.selectbox("Algorithm", ["Show All", "✨ Hybrid", "👥 Community", "🎭 AI DNA"], label_visibility="collapsed")

# --- RE-DESIGNED HELPER FUNCTION ---
def render_movie_cards(recommendations, score_column):
    # CRITICAL FIX: The HTML string is built continuously on one line to prevent Streamlit 
    # from interpreting indentation as a Markdown Code Block.
    html_content = '<div class="scroll-container">'
    
    if score_column == 'CB_Score': match_reason = "Based on DNA"
    elif score_column == 'CF_Score': match_reason = "Community Pick"
    elif score_column == 'Hybrid_Score': match_reason = "Best Match"
    else: match_reason = "Global Search"
    
    for i, (_, row) in enumerate(recommendations.iterrows()):
        poster_url, overview, movie_link = fetch_movie_details(row['title'])
        score = row.get(score_column, 85)
        
        # Single-line append prevents the text block bug
        html_content += f'<div class="movie-card"><img src="{poster_url}" class="movie-poster"><div class="card-overlay"><div class="movie-title">{row["title"]}</div><div class="match-score">{score:.0f}% Match <span class="reason-text">• {match_reason}</span></div><a href="{movie_link}" target="_blank" class="watch-btn">DETAILS</a></div></div>'
        
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

# --- MAIN UI LOGIC ---
st.divider()

if search_query:
    with st.spinner('Scanning database...'):
        query_vec = tfidf.transform([search_query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_match_idx = sim_scores.argmax()
        best_score = sim_scores[best_match_idx]
        
        if best_score > 0.1: 
            selected_movie = movies.iloc[best_match_idx]['title']
            
            if selected_display in ["Show All", "✨ Hybrid"]:
                st.markdown('<p class="category-header">Recommended for you</p>', unsafe_allow_html=True)
                render_movie_cards(get_hybrid_recs(selected_movie), 'Hybrid_Score')
                    
            if selected_display in ["Show All", "👥 Community"]:
                st.markdown('<p class="category-header">Customers also watched</p>', unsafe_allow_html=True)
                render_movie_cards(get_community_recs(selected_movie), 'CF_Score')
            
            if selected_display in ["Show All", "🎭 AI DNA"]:
                st.markdown('<p class="category-header">More like this</p>', unsafe_allow_html=True)
                render_movie_cards(get_content_based_recs(selected_movie), 'CB_Score')
                    
        else:
            topic_results = search_tmdb_topic(search_query)
            if topic_results:
                st.markdown(f'<p class="category-header">Results for "{search_query}"</p>', unsafe_allow_html=True)
                render_movie_cards(pd.DataFrame(topic_results), 'score')
            else:
                st.warning("No results found.")
