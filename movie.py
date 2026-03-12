import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from movie_logic import (
    movies, cosine_sim, tfidf, tfidf_matrix,
    fetch_movie_details, search_tmdb_topic,
    get_content_based_recs, get_community_recs, get_hybrid_recs
)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CineMatch Pro", page_icon="🍿", layout="wide")

# --- CINEMATIC UI (PRIME VIDEO INSPIRED) ---
st.markdown("""
    <style>
    /* 1. Global Dark Theme Override */
    .stApp {
        background: radial-gradient(circle at top, #1a242f 0%, #0f171e 100%) !important;
        color: #ffffff !important;
    }

    /* 2. Prime Video Style Typography */
    .main-title { 
        font-size: 3.5rem; 
        font-weight: 800; 
        color: #ffffff; 
        text-align: left; 
        margin-bottom: -10px;
        letter-spacing: -1.5px;
    }
    .main-title-blue { color: #00A8E1; }
    .sub-title { 
        font-size: 1rem; 
        color: #8197a4; 
        text-align: left; 
        margin-bottom: 2rem; 
    }

    /* 3. Section Headers */
    .category-header { 
        font-size: 1.4rem; 
        color: #ffffff !important; 
        font-weight: 700; 
        margin-top: 2rem; 
        margin-bottom: 1rem; 
    }

    /* 4. Horizontal Scroll Container (Hidden Scrollbar) */
    .scroll-container { 
        display: flex; 
        flex-wrap: nowrap; 
        overflow-x: auto; 
        gap: 16px; 
        padding: 20px 0px; 
        scroll-behavior: smooth;
    }
    .scroll-container::-webkit-scrollbar { display: none; }

    /* 5. The Prime Card: Poster Focus + Zoom Effect */
    .movie-card { 
        flex: 0 0 220px; 
        background-color: #1a242f; 
        border-radius: 8px; 
        position: relative;
        transition: transform 0.4s cubic-bezier(0.2, 1, 0.3, 1), box-shadow 0.4s;
        cursor: pointer;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .movie-card:hover { 
        transform: scale(1.1); 
        z-index: 100;
        box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        border-color: #00A8E1;
    }

    .movie-poster { 
        width: 100%; 
        aspect-ratio: 2/3;
        object-fit: cover;
    }

    /* Info overlay that appears/highlights on hover */
    .card-overlay {
        padding: 12px;
        background: linear-gradient(to top, #0f171e 0%, transparent 100%);
    }

    .movie-title { 
        font-size: 0.95rem; 
        color: #ffffff !important; 
        font-weight: 700; 
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
    }

    .match-score { 
        color: #46d369 !important; /* Prime/Netflix Style Green Match */
        font-weight: 700; 
        font-size: 0.85rem; 
    }

    .watch-btn { 
        background-color: #00A8E1; 
        color: #ffffff !important; 
        padding: 8px; 
        border-radius: 4px; 
        text-decoration: none; 
        font-weight: 700; 
        display: block; 
        text-align: center;
        font-size: 0.8rem;
        margin-top: 10px;
    }

    /* 6. Form elements styling to match dark theme */
    div[data-baseweb="input"] { background-color: #1a242f !important; border: 1px solid #303b44 !important; }
    input { color: white !important; }
    .stSelectbox div { background-color: #1a242f !important; color: white !important; }
    
    hr { border-color: rgba(255,255,255,0.1) !important; }
    </style>
""", unsafe_allow_html=True)

# --- RE-DESIGNED HELPER FUNCTION ---
def render_movie_cards(recommendations, score_column):
    html_content = '<div class="scroll-container">'
    
    if score_column == 'CB_Score': match_reason = "Based on DNA"
    elif score_column == 'CF_Score': match_reason = "Community Pick"
    elif score_column == 'Hybrid_Score': match_reason = "Best Match"
    else: match_reason = "Global Search"
    
    for i, (_, row) in enumerate(recommendations.iterrows()):
        poster_url, overview, movie_link = fetch_movie_details(row['title'])
        score = row.get(score_column, 85)
        
        html_content += f"""
        <div class="movie-card">
            <img src="{poster_url}" class="movie-poster">
            <div class="card-overlay">
                <div class="match-score">{score:.0f}% Match <span style="color:#8197a4; font-weight:400; font-size:0.7rem;">• {match_reason}</span></div>
                <div class="movie-title">{row['title']}</div>
                <a href="{movie_link}" target="_blank" class="watch-btn">DETAILS</a>
            </div>
        </div>
        """
        
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

# --- MAIN UI ---
st.markdown('<p class="main-title">CineMatch<span class="main-title-blue">Pro</span></p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-curated recommendations with Prime Video precision.</p>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
search_query = col1.text_input("Search", placeholder="Search for movies, actors, or themes...", label_visibility="collapsed")
selected_display = col2.selectbox("Algorithm", ["Show All", "✨ Hybrid", "👥 Community", "🎭 AI DNA"], label_visibility="collapsed")

st.divider()

if search_query:
    with st.spinner('Scanning database...'):
        query_vec = tfidf.transform([search_query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_match_idx = sim_scores.argmax()
        best_score = sim_scores[best_match_idx]
        
        if best_score > 0.1: # Threshold adjusted for NLP
            selected_movie = movies.iloc[best_match_idx]['title']
            
            # Row 1: Hybrid
            if selected_display in ["Show All", "✨ Hybrid"]:
                st.markdown('<p class="category-header">Recommended for you</p>', unsafe_allow_html=True)
                render_movie_cards(get_hybrid_recs(selected_movie), 'Hybrid_Score')
                    
            # Row 2: Collaborative
            if selected_display in ["Show All", "👥 Community"]:
                st.markdown('<p class="category-header">Customers also watched</p>', unsafe_allow_html=True)
                render_movie_cards(get_community_recs(selected_movie), 'CF_Score')
            
            # Row 3: Content
            if selected_display in ["Show All", "🎭 AI DNA"]:
                st.markdown('<p class="category-header">More like this</p>', unsafe_allow_html=True)
                render_movie_cards(get_content_based_recs(selected_movie), 'CB_Score')
                    
        else:
            # Fallback to TMDB Topic Search
            topic_results = search_tmdb_topic(search_query)
            if topic_results:
                st.markdown(f'<p class="category-header">Results for "{search_query}"</p>', unsafe_allow_html=True)
                render_movie_cards(pd.DataFrame(topic_results), 'score')
            else:
                st.warning("No results found.")
else:
    # Featured Hero Placeholder (Optional)
    st.info("Start typing to see cinematic recommendations.")
