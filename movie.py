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
st.set_page_config(page_title="CineMatch Pro", page_icon="🍿", layout="wide")

# Setting up the Dark Netflix-Style UI
st.markdown("""
    <style>
    /* Dark background for the main app */
    .stApp { background-color: #0c0c0c !important; color: #ffffff !important; }
    
    /* Hide top padding and default Streamlit elements */
    .block-container { padding-top: 2rem !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    /* Main Logo / Title */
    .main-logo { font-size: 2.2rem; font-weight: 900; color: #E50914; margin-bottom: 0px; letter-spacing: 1px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;}
    .main-logo span { color: #ffffff; font-weight: 300; }
    
    /* Category titles */
    .category-header { font-size: 1.2rem; color: #e5e5e5; font-weight: 600; margin-top: 2rem; margin-bottom: 10px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;}

    /* Horizontal Scroll Container */
    .scroll-container { display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 15px; padding: 10px 0px 30px 0px; scroll-behavior: smooth; }
    .scroll-container::-webkit-scrollbar { height: 6px; }
    .scroll-container::-webkit-scrollbar-track { background: transparent; }
    .scroll-container::-webkit-scrollbar-thumb { background: #333333; border-radius: 10px; }
    .scroll-container::-webkit-scrollbar-thumb:hover { background: #E50914; }

    /* Minimalist Movie Cards */
    .movie-card { 
        flex: 0 0 160px; /* Width of the cards */
        display: flex; 
        flex-direction: column; 
        text-decoration: none; /* For the clickable link */
        cursor: pointer;
    }
    
    /* Poster Container & Hover Effect */
    .poster-container {
        width: 100%;
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    }
    .poster-container img { 
        width: 100%; 
        display: block;
        transition: transform 0.3s ease; 
    }
    .movie-card:hover .poster-container img { 
        transform: scale(1.08); 
    }

    /* Movie Text Meta */
    .movie-title { 
        font-size: 0.95rem; 
        color: #ffffff; 
        font-weight: 600; 
        margin-bottom: 4px; 
        white-space: nowrap; 
        overflow: hidden; 
        text-overflow: ellipsis; 
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .movie-meta { 
        font-size: 0.8rem; 
        color: #808080; 
        display: flex; 
        align-items: center; 
        gap: 8px;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .movie-meta span.score { color: #46d369; font-weight: 600; } /* Netflix green match text */
    .movie-meta span.star { color: #e5a00d; } /* Yellow star rating */
    
    /* Style Streamlit Inputs to blend into dark mode */
    .stTextInput input { background-color: #141414 !important; color: white !important; border: 1px solid #333 !important; border-radius: 4px !important; }
    .stTextInput input:focus { border-color: #E50914 !important; box-shadow: none !important; }
    .stSelectbox div[data-baseweb="select"] { background-color: #141414 !important; color: white !important; border: 1px solid #333 !important; }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTION TO RENDER UI CARDS ---
def render_movie_cards(recommendations, score_column):
    html_content = '<div class="scroll-container">'
    
    for i, (_, row) in enumerate(recommendations.iterrows()):
        poster_url, overview, movie_link = fetch_movie_details(row['title'])
        
        # Format the score out of 10 for the "Star" rating look from your image, 
        # or as a percentage for the "Match" look.
        score = row.get(score_column, 85)
        formatted_score = f"{score/10:.1f}" if score > 10 else f"{score:.1f}"
        percentage_match = f"{score:.0f}% Match" if score > 10 else f"{score*10:.0f}% Match"
        
        # Generates a card wrapped in an anchor tag so the whole thing is clickable
        html_content += f"""
        <a href="{movie_link}" target="_blank" class="movie-card" style="text-decoration: none;">
            <div class="poster-container">
                <img src="{poster_url}" alt="{row['title']}">
            </div>
            <div class="movie-title">{row['title']}</div>
            <div class="movie-meta">
                <span class="score">{percentage_match}</span>
                <span><span class="star">★</span> {formatted_score}</span>
            </div>
        </a>
        """
        
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

# --- MAIN UI LAYOUT ---

# Top Navigation / Logo mimicking the first image
st.markdown('<p class="main-logo">CINEMATCH<span>PRO</span></p>', unsafe_allow_html=True)
st.write("") # small spacer

# Search and Model selection
col1, col2 = st.columns([3, 1])
search_query = col1.text_input("Search", placeholder="Titles, people, genres...", label_visibility="collapsed")
selected_display = col2.selectbox("Categories", ["All Picks", "Top Picks (Hybrid)", "Trending Now", "AI Similar"], label_visibility="collapsed")

# Results Section
if search_query:
    with st.spinner('Curating...'):
        
        # 1. UNIVERSAL SEARCH (NLP)
        query_vec = tfidf.transform([search_query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        best_match_idx = sim_scores.argmax()
        best_score = sim_scores[best_match_idx]
        
        # 2. DECISION LOGIC: Known Movie or Topic?
        if best_score > 0:
            selected_movie = movies.iloc[best_match_idx]['title']
            
            # --- RENDER RESULTS ROWS based on user selection ---
            
            # Row 1: Hybrid Top Picks
            if selected_display in ["All Picks", "Top Picks (Hybrid)"]:
                st.markdown('<p class="category-header">Top Picks for You</p>', unsafe_allow_html=True)
                render_movie_cards(get_hybrid_recs(selected_movie), 'Hybrid_Score')
                    
            # Row 2: Collaborative (Community-Based) Picks
            if selected_display in ["All Picks", "Trending Now"]:
                st.markdown('<p class="category-header">Trending Now</p>', unsafe_allow_html=True)
                render_movie_cards(get_community_recs(selected_movie), 'CF_Score')
            
            # Row 3: Content-Based (AI DNA-Based) Picks
            if selected_display in ["All Picks", "AI Similar"]:
                st.markdown('<p class="category-header">Because you searched for ' + selected_movie + '</p>', unsafe_allow_html=True)
                render_movie_cards(get_content_based_recs(selected_movie), 'CB_Score')
                    
        # 3. GLOBAL FALLBACK: Search TMDB topic if no local match is found
        else:
            st.markdown(f'<p class="category-header">Search Results for "{search_query}"</p>', unsafe_allow_html=True)
            topic_results = search_tmdb_topic(search_query)
            
            if topic_results:
                topic_df = pd.DataFrame(topic_results)
                render_movie_cards(topic_df, 'score')
            else:
                st.warning("No movies found for that search.")

else:
    # Default Hero/Landing state when empty
    st.markdown('<p class="category-header" style="text-align:center; margin-top:10vh; color:#666;">Search for a movie, genre, or keyword to get started.</p>', unsafe_allow_html=True)
