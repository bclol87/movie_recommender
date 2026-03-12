import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- MAGIC STEP: Import our backend logic from the new file! ---
from movie_logic import (
    movies, tfidf, tfidf_matrix, 
    fetch_movie_details, search_tmdb_topic, 
    get_content_based_recs, get_community_recs, get_hybrid_recs
)

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="CineMatch Pro", page_icon="🍿", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0b0b0b !important; color: #ffffff !important; }
    .main-title { font-size: 4rem; font-weight: 900; margin-bottom: 0px; text-align: center; background: linear-gradient(90deg, #E50914, #ff7b00); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .sub-title { color: #aaaaaa; font-size: 1.2rem; margin-bottom: 2rem; font-weight: 400; text-align: center; }
    .category-header { font-size: 1.5rem; color: #ffffff !important; font-weight: bold; margin-top: 2rem; margin-bottom: 1rem; border-left: 5px solid #E50914; padding-left: 10px; }
    .scroll-container { display: flex; flex-wrap: nowrap; overflow-x: auto; overflow-y: hidden; gap: 20px; padding: 10px 0px 20px 0px; scroll-behavior: smooth; }
    .scroll-container::-webkit-scrollbar { height: 12px; }
    .scroll-container::-webkit-scrollbar-track { background: #181818; border-radius: 10px; }
    .scroll-container::-webkit-scrollbar-thumb { background: #E50914; border-radius: 10px; border: 2px solid #181818; }
    .scroll-container::-webkit-scrollbar-thumb:hover { background: #ff0a16; }
    
    .movie-card { flex: 0 0 240px; background: #181818; padding: 15px; border-radius: 10px; text-align: center; height: 600px; display: flex; flex-direction: column; border: 1px solid #333333; transition: transform 0.2s, box-shadow 0.2s; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
    .movie-card:hover { transform: translateY(-5px); border-color: #E50914; box-shadow: 0 8px 20px rgba(229, 9, 20, 0.4); }
    .movie-poster { width: 100%; height: 320px; object-fit: cover; border-radius: 8px; margin-bottom: 12px; }
    .movie-title { font-size: 1.1rem; color: #ffffff; font-weight: bold; margin-bottom: 5px; min-height: 2.8rem; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .match-score { color: #46d369; font-weight: bold; font-size: 1rem; margin-bottom: 5px; min-height: 1.2rem; }
    .movie-overview { font-size: 0.85rem; color: #cccccc; text-align: left; margin-bottom: 15px; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; flex-grow: 1; }
    .watch-btn { background-color: #E50914; color: white !important; padding: 8px; border-radius: 4px; text-decoration: none; font-weight: bold; display: block; width: 100%; margin-top: auto; }
    .watch-btn:hover { background-color: #f40612; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTION TO RENDER UI CARDS ---
def render_movie_cards(recommendations, score_column):
    html_content = '<div class="scroll-container">'
    
    if score_column == 'CB_Score': match_reason = "🧬 Matches Actors, Director & Plot"
    elif score_column == 'CF_Score': match_reason = "⭐ Global Community Rating"
    elif score_column == 'Hybrid_Score': match_reason = "✨ AI DNA + Global Rating"
    else: match_reason = "🌐 TMDB Search Result"
    
    for i, (_, row) in enumerate(recommendations.iterrows()):
        poster_url, overview, movie_link = fetch_movie_details(row['title'])
        score = row.get(score_column, 85)
        
        html_content += f"""<div class="movie-card">
<img src="{poster_url}" class="movie-poster" alt="poster">
<div class="movie-title">{row['title']}</div>
<div class="match-score">{score:.0f}% Match</div>
<div style="font-size: 0.75rem; color: #888888; margin-top: -8px; margin-bottom: 10px;">{match_reason}</div>
<div class="movie-overview">{overview}</div>
<a href="{movie_link}" target="_blank" class="watch-btn">View Details</a>
</div>"""
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

# --- MAIN UI LAYOUT ---
st.markdown('<p class="main-title">CineMatch Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by Content-Based & Community Algorithms.</p>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
search_query = col1.text_input("Search", placeholder="Type a movie title or topic (e.g., 'race car')...", label_visibility="collapsed")
selected_display = col2.selectbox("Choose Model:", ["Show All Rows", "✨ Top Picks (Hybrid)", "👥 Community Picks", "🎭 AI Similar (Content-Based)"], label_visibility="collapsed")

st.divider()

if search_query:
    with st.spinner('Curating dashboard...'):
        query_vec = tfidf.transform([search_query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        best_match_idx = sim_scores.argmax()
        best_score = sim_scores[best_match_idx]
        
        if best_score > 0:
            selected_movie = movies.iloc[best_match_idx]['title']
            movie_type = movies.iloc[best_match_idx]['genres_clean'].replace(" ", ", ")
            
            st.success(f"🧠 NLP AI analyzed your search and selected: **{selected_movie}**")
            st.info(f"🏷️ **Movie Type:** {movie_type}")
            
            if selected_display in ["Show All Rows", "✨ Top Picks (Hybrid)"]:
                st.markdown('<p class="category-header">✨ Hybrid Top Picks</p>', unsafe_allow_html=True)
                render_movie_cards(get_hybrid_recs(selected_movie), 'Hybrid_Score')
                    
            if selected_display in ["Show All Rows", "👥 Community Picks"]:
                st.markdown('<p class="category-header">👥 Community Favorites</p>', unsafe_allow_html=True)
                render_movie_cards(get_community_recs(selected_movie), 'CF_Score')
            
            if selected_display in ["Show All Rows", "🎭 AI Similar (Content-Based)"]:
                st.markdown('<p class="category-header">🎭 Content Similarity</p>', unsafe_allow_html=True)
                render_movie_cards(get_content_based_recs(selected_movie), 'CB_Score')
                    
        else:
            st.info(f"🌐 No local matches. Searching global TMDB database for: **'{search_query}'**")
            topic_results = search_tmdb_topic(search_query)
            
            if topic_results:
                topic_df = pd.DataFrame(topic_results)
                render_movie_cards(topic_df, 'score')
            else:
                st.warning("No movies found for that topic.")
else:
    st.info("👆 Type a movie name (e.g., 'Iron Man') OR a description (e.g., 'race car') to run your AI models!")
