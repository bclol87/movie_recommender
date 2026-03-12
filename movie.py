import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --- MAGIC STEP: Import our functions from our new logic file ---
# Python must find movie_logic.py in the same directory.
from movie_logic import (
    movies, cosine_sim, tfidf, tfidf_matrix,
    fetch_movie_details, search_tmdb_topic,
    get_content_based_recs, get_community_recs, get_hybrid_recs
)

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="CineMatch Pro", page_icon="🍿", layout="wide")

# Setting up the white background with BLACK movie cards (as requested)
st.markdown("""
    <style>
    /* Force white background for the main app */
    .stApp { background-color: #ffffff !important; color: #000000 !important; }
    
    /* Netflix-Style Typography in Red/Orange */
    .main-title { font-size: 3.5rem; font-weight: 800; color: #E50914; text-align: center; margin-bottom: 0px; }
    .main-title-orange { color: #ff7b00; }
    .sub-title { font-size: 1.1rem; color: #555555; text-align: center; margin-bottom: 2rem; font-weight: 400; }
    
    /* Category titles: Red text with left red bar */
    .category-header { font-size: 1.6rem; color: #E50914 !important; font-weight: 700; margin-top: 2rem; margin-bottom: 1.5rem; border-left: 5px solid #E50914; padding-left: 10px; }

    /* The main grid for movie picks, stylized like the references */
    .scroll-container { display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 20px; padding: 10px 0px; scroll-behavior: smooth; }
    .scroll-container::-webkit-scrollbar { height: 12px; }
    .scroll-container::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
    .scroll-container::-webkit-scrollbar-thumb { background: #cccccc; border-radius: 10px; }
    .scroll-container::-webkit-scrollbar-thumb:hover { background: #E50914; }

    /* Movie Cards: Rounded, with a black background and white text */
    .movie-card { flex: 0 0 240px; background-color: #121212; padding: 15px; border-radius: 10px; text-align: center; height: 620px; display: flex; flex-direction: column; transition: transform 0.2s, box-shadow 0.2s; box-shadow: 0 4px 8px rgba(0,0,0,0.3); border: 1px solid #333333; }
    .movie-card:hover { transform: translateY(-5px); border-color: #E50914; box-shadow: 0 8px 16px rgba(229, 9, 20, 0.3); }
    .movie-card img { width: 100%; border-radius: 8px; margin-bottom: 15px; }
    .movie-title { font-size: 1.1rem; color: #ffffff !important; font-weight: 700; margin-bottom: 8px; min-height: 2.8rem; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
    
    /* Match Score in Red as per reference */
    .match-score { color: #E50914 !important; font-weight: 800; font-size: 1.2rem; margin-bottom: 5px; }
    
    /* Summary and detail text in gray for readability */
    .movie-overview { font-size: 0.85rem; color: #bbbbbb; text-align: left; margin-bottom: 15px; flex-grow: 1; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; }
    
    /* View Details button in pure Red */
    .watch-btn { background-color: #E50914; color: #ffffff !important; padding: 10px; border-radius: 6px; text-decoration: none; font-weight: 700; display: block; width: 100%; margin-top: auto; border: none; }
    .watch-btn:hover { background-color: #f40612; }

    /* Forcing black text on other elements */
    .stTextInput input { color: black !important; }
    .stSelectbox div { color: black !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTION TO RENDER UI CARDS ---
# This remains in movie.py as it is purely presentation logic
def render_movie_cards(recommendations, score_column):
    html_content = '<div class="scroll-container">'
    
    # Explainable AI Logic: Determine the reason for the match
    if score_column == 'CB_Score': match_reason = "🧬 Matches Actors, Director & Plot"
    elif score_column == 'CF_Score': match_reason = "⭐ Global Community Rating"
    elif score_column == 'Hybrid_Score': match_reason = "✨ AI DNA + Global Rating"
    else: match_reason = "🌐 TMDB Search Result"
    
    for i, (_, row) in enumerate(recommendations.iterrows()):
        poster_url, overview, movie_link = fetch_movie_details(row['title'])
        score = row.get(score_column, 85) # Fallback to 85% if no score exists
        
        html_content += f"""<div class="movie-card">
<img src="{poster_url}" class="movie-poster" alt="poster">
<div class="movie-title">{row['title']}</div>
<div class="match-score">{score:.0f}% Match</div>
<div style="font-size: 0.75rem; color: #bbbbbb; margin-top: -8px; margin-bottom: 10px;">{match_reason}</div>
<div class="movie-overview">{overview}</div>
<a href="{movie_link}" target="_blank" class="watch-btn">View Details</a>
</div>"""
        
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

# --- MAIN UI LAYOUT ---
# Header
st.markdown('<p class="main-title">CineMatch<span class="main-title-orange">Pro</span></p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by AI-Based (Content) and Community-Based (Collaborative) Algorithms.</p>', unsafe_allow_html=True)

# Search and Model selection
col1, col2 = st.columns([3, 1])
search_query = col1.text_input("Search", placeholder="Type a movie title or description (e.g., 'race car')...", label_visibility="collapsed")
selected_display = col2.selectbox("Choose Model:", ["Show All Rows", "✨ Top Picks (Hybrid)", "👥 Community Picks", "🎭 AI Similar (Content-Based)"], label_visibility="collapsed")

st.divider()

# Results Section
if search_query:
    with st.spinner('CineMatchPro Pro activated... curating your movie dashboard...'):
        
        # 1. UNIVERSAL SEARCH (NLP) - deciding if it's a known title or plot description
        # We transform user input using our tfidf tool to get a vector
        query_vec = tfidf.transform([search_query])
        # Compare vector against our entire local database matrix
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Find index and score of best local match
        best_match_idx = sim_scores.argmax()
        best_score = sim_scores[best_match_idx]
        
        # SQA Ghost Bug fix is now handled by the Tfidf upgrade (step 1 above).
        # We search with no difflib correction, letting the NLP AI decide the match.
        
        # 2. DECISION LOGIC: Known Movie or Topic?
        # If the local similarity score is high (above ~0.15 for title match), treat as known movie
        if best_score > 0:
            selected_movie = movies.iloc[best_match_idx]['title']
            movie_type = movies.iloc[best_match_idx]['genres_clean'].replace(" ", ", ")

            st.success(f"🧠 Local NLP AI is running models for movie: **{selected_movie}**")
            st.info(f"🏷️ **Movie Type:** {movie_type}")
            
            # --- RENDER RESULTS ROWS based on user selection ---
            
            # Row 1: Hybrid Top Picks
            if selected_display in ["Show All Rows", "✨ Top Picks (Hybrid)"]:
                st.markdown('<p class="category-header">✨ Hybrid Top Picks</p>', unsafe_allow_html=True)
                render_movie_cards(get_hybrid_recs(selected_movie), 'Hybrid_Score')
                    
            # Row 2: Collaborative (Community-Based) Picks
            if selected_display in ["Show All Rows", "👥 Community Picks"]:
                st.markdown('<p class="category-header">👥 Community Favorites</p>', unsafe_allow_html=True)
                render_movie_cards(get_community_recs(selected_movie), 'CF_Score')
            
            # Row 3: Content-Based (AI DNA-Based) Picks
            if selected_display in ["Show All Rows", "🎭 AI Similar (Content-Based)"]:
                st.markdown('<p class="category-header">🎭 AI Similarity</p>', unsafe_allow_html=True)
                render_movie_cards(get_content_based_recs(selected_movie), 'CB_Score')
                    
        # 3. GLOBAL FALLBACK: Search TMDB topic if no local match is found
        else:
            st.info(f"🌐 Query analysis suggests a topic search. Checking global TMDB library for: **'{search_query}'**")
            topic_results = search_tmdb_topic(search_query)
            
            if topic_results:
                st.success(f"🎯 Found movies matching the global topic: **'{search_query}'**")
                # Convert list of dicts to DataFrame to work with render function
                topic_df = pd.DataFrame(topic_results)
                render_movie_cards(topic_df, 'score')
            else:
                st.warning("No movies found for that topic.")

# Welcome/Default message
else:
    st.info("👆 Type a movie name (e.g., 'X-Men') OR a description (e.g., 'mutant striker') to start your recommendations dashboard!")
