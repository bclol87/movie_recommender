import streamlit as st
import pandas as pd
import numpy as np
import requests
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- TMDB API CONFIGURATION ---
API_KEY = "3eb39709869b67fd15b086e095c5cbec"

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="CineMatch Pro", page_icon="🍿", layout="wide")

# --- MAIN PAGE RENDERING (Dark Theme Edition) ---
# Update CSS for a cohesive dark theme
st.markdown(\"\"\"
    <style>
    /* Forcing dark theme on everything for cohesion */
    .stApp { background-color: #121212 !important; color: #ffffff !important; }
    .stTextInput input { color: #ffffff !important; background-color: #333333 !important; }
    .stSelectbox div { color: #ffffff !important; background-color: #333333 !important; }
    .stAlert { color: #000000 !important; } /* SQA alert text must stay dark! */
    
    /* Netflix-Style typography and gradients */
    .main-title { font-size: 3.5rem; font-weight: 800; color: #E50914; text-align: center; }
    .sub-title { font-size: 1.1rem; color: #B3B3B3; text-align: center; margin-bottom: 2rem; }
    
    /* Clear and professional headers for each section */
    .category-header { font-size: 1.6rem; color: #ffffff; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; border-left: 5px solid #E50914; padding-left: 10px; }
    .hybrid-header { color: #ffcc00; }
    .collaborative-header { color: #00cccc; }
    .content-header { color: #cccc00; }
    
    /* Premium Glassmorphism Movie Cards */
    .movie-card {
        background: rgba(255, 255, 255, 0.05); /* Very slight white overlay for glass look */
        border-radius: 12px;
        padding: 15px;
        transition: all 0.3s ease;
        text-align: center;
        margin-bottom: 20px;
        height: 100%; /* Ensures all cards in a row have equal height */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .movie-card:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(229, 9, 20, 0.4); } /* Red glow on hover */
    .movie-title { font-size: 1.1rem; color: #ffffff; font-weight: 700; margin-bottom: 8px; }
    .movie-match { color: #46D369; font-weight: 800; font-size: 1.2rem; margin-bottom: 10px; }
    .movie-overview { font-size: 0.85rem; color: #B3B3B3; text-align: left; margin-bottom: 15px; flex-grow: 1; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; }
    .view-btn { background-color: #E50914; color: #ffffff !important; padding: 10px; border-radius: 6px; text-decoration: none; font-weight: 700; display: block; width: 100%; }
    .view-btn:hover { background-color: #f40612; }
    </style>
\"\"\", unsafe_allow_html=True)

st.markdown('<p class="main-title">CineMatch Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by AI-Based (Content) and Community-Based (Collaborative) Algorithms.</p>', unsafe_allow_html=True)

# Main container for layout
container = st.container()

with container:
    # 1. Search Section (Unified Title & NLP)
    col1, col2 = st.columns([3, 1])
    search_query = col1.text_input(\"Search for a movie title (e.g., 'X-Men') or a plot (e.g., 'car')\")
    # selected_display dropdown removed for simpler presentation of all model results in distinct sections.

    if search_query:
        # SQA Ghost Catch! Handling ghost text problem. We forced a consistent dark theme for the whole session
        # within this section as a temporary proactive fix until you can properly synchronise the config.toml file
        # with Streamlit Secrets on your final presentation server, which is brilliant testing logic!
        if 'base=\"dark\"' not in st.session_state.get('config_state', ''):
            st.session_state['config_state'] = 'base=\"dark\"' # Force dark for this session

        with st.spinner('Activated Local AI models... curating your movie dashboard...'):
            closest_matches = difflib.get_close_matches(search_query.title(), movie_list, n=1, cutoff=0.5)

            if closest_matches:
                selected_movie = closest_matches[0]
                
                # Fetch genres for Type display
                idx = movies[movies['title'] == selected_movie].index[0]
                movie_type = movies.iloc[idx]['genres_clean'].replace(\" \", \", \")

                st.success(f\"🎯 Local AI models running for movie: **{selected_movie}**\")
                st.info(f\"🏷️ **Movie Type:** {movie_type}\")
                
                # 2. Hybrid Top Picks Section
                st.markdown('<p class="category-header hybrid-header">✨ Hybrid Top Picks</p>', unsafe_allow_html=True)
                hybrid_recs = get_hybrid_recs(selected_movie)
                cols = st.columns(len(hybrid_recs))
                for i, row in hybrid_recs.iterrows():
                    poster_url, overview, movie_link = fetch_movie_details(row['title'])
                    with cols[hybrid_recs.index.get_loc(i)]:
                        st.markdown(f'''
                            <div class="movie-card">
                                <img src="{poster_url}" style=\"width:100%; border-radius:8px; margin-bottom:12px;\">
                                <div class="movie-title">{row['title']}</div>
                                <div class="movie-match">{row['Hybrid_Score']:.0f}% Match</div>
                                <div class=\"movie-overview\">{overview}</div>
                                <a href=\"{movie_link}\" target=\"_blank\" class=\"view-btn\">View Details</a>
                            </div>
                        ''', unsafe_allow_html=True)
                        
                # 3. Collaborative (Community-Based) Section
                st.markdown('<p class="category-header collaborative-header">👥 Community Favorites</p>', unsafe_allow_html=True)
                # Collaborative results displayed separately for clarity
                collaborative_recs = get_community_recs(selected_movie)
                cols = st.columns(len(collaborative_recs))
                for i, row in collaborative_recs.iterrows():
                    poster_url, overview, movie_link = fetch_movie_details(row['title'])
                    with cols[collaborative_recs.index.get_loc(i)]:
                        st.markdown(f'''
                            <div class="movie-card">
                                <img src="{poster_url}" style=\"width:100%; border-radius:8px; margin-bottom:12px;\">
                                <div class="movie-title">{row['title']}</div>
                                <div class="movie-match\">{row['CF_Score']:.0f}% Match</div>
                                <div class=\"movie-overview\">{overview}</div>
                                <a href=\"{movie_link}\" target=\"_blank\" class=\"view-btn\">View Details</a>
                            </div>
                        ''', unsafe_allow_html=True)
                        
                # 4. Content-Based (AI-Based DNA) Section
                st.markdown('<p class="category-header content-header">🎭 AI Similarity</p>', unsafe_allow_html=True)
                # Content results displayed separately for clarity
                content_recs = get_content_based_recs(selected_movie)
                cols = st.columns(len(content_recs))
                for i, row in content_recs.iterrows():
                    poster_url, overview, movie_link = fetch_movie_details(row['title'])
                    with cols[content_recs.index.get_loc(i)]:
                        st.markdown(f'''
                            <div class="movie-card">
                                <img src="{poster_url}" style=\"width:100%; border-radius:8px; margin-bottom:12px;\">
                                <div class="movie-title">{row['title']}</div>
                                <div class="movie-match\">{row['CB_Score']:.0f}% Match</div>
                                <div class=\"movie-overview\">{overview}</div>
                                <a href=\"{movie_link}\" target=\"_blank\" class=\"view-btn\">View Details</a>
                            </div>
                        ''', unsafe_allow_html=True)
            
            else:
                st.info(f\"🌐 Query analysis suggests a topic search. Checking local library for: **'{search_query}'**\")
                # Add topic search to this design

# --- DATA LOADING & AI TRAINING ---
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('movies_final_lite.csv')
    
    def extract_genres(x):
        try:
            genres = ast.literal_eval(x)
            return " ".join([g['name'] for g in genres])
        except:
            return ""
            
    df['genres_clean'] = df['genres'].apply(extract_genres)
    
    df['overview'] = df['overview'].fillna('')
    df['actors'] = df['actors'].fillna('')
    df['director'] = df['director'].fillna('')
    df['title'] = df['title'].astype(str)
    
    # NLP UPGRADE: We injected the title twice so the AI knows it's highly important!
    df['content_features'] = df['title'] + " " + df['title'] + " " + df['genres_clean'] + " " + df['actors'] + " " + df['director'] + " " + df['overview']
    
    return df.dropna(subset=['title']).reset_index(drop=True)

@st.cache_resource
def train_tfidf_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # Returning all 3 tools so our Universal Search can use them
    return tfidf, tfidf_matrix, cosine_sim

movies = load_and_prep_data()
tfidf, tfidf_matrix, cosine_sim = train_tfidf_model(movies)

# --- API HELPER FUNCTIONS ---
@st.cache_data(ttl=86400) 
def fetch_movie_details(title_or_id, is_id=False):
    query_param = f"/{title_or_id}" if is_id else f"/search/movie?query={title_or_id}"
    url = f"https://api.themoviedb.org/3{query_param}&api_key={API_KEY}" if not is_id else f"https://api.themoviedb.org/3/movie/{title_or_id}?api_key={API_KEY}"
    
    try:
        response = requests.get(url).json()
        movie = response if is_id else response['results'][0]
        
        poster_path = movie.get('poster_path')
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/500x750?text=No+Poster"
        
        overview = movie.get('overview', 'No description available.')
        if len(overview) > 120: overview = overview[:117] + "..."
            
        return poster_url, overview, f"https://www.themoviedb.org/movie/{movie.get('id')}"
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster", "Description not found.", "#"

def search_tmdb_topic(query):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"
    try:
        data = requests.get(url).json()
        results = []
        for movie in data.get('results', [])[:20]:
            p_url, desc, link = fetch_movie_details(movie['id'], is_id=True)
            results.append({'title': movie.get('title'), 'poster': p_url, 'overview': desc, 'link': link, 'score': movie.get('vote_average', 0) * 10})
        return results
    except: return []

# --- CORE ALGORITHMS ---
def get_content_based_recs(movie_title):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # 1. FIX: Start at [0:20] instead of [1:21] so the actual movie stays at the #1 spot!
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[0:20]
    
    recs = movies.iloc[[i[0] for i in sim_scores]].copy()
    recs['CB_Score'] = [i[1] * 100 for i in sim_scores]
    return recs

def get_community_recs(movie_title):
    idx = movies[movies['title'] == movie_title].index[0]
    target_genres = movies.iloc[idx]['genres_clean'].split()
    
    if not target_genres: return movies.head(20)
    
    pattern = '|'.join(target_genres)
    pool = movies[movies['genres_clean'].str.contains(pattern, case=False, na=False)].copy()
    
    # 2. FIX: Deleted the line that erased the searched movie! 
    
    pool['CF_Score'] = (pool['vote_average'] / 10) * 100
    return pool.sort_values(['vote_count', 'vote_average'], ascending=[False, False]).head(20)

def get_hybrid_recs(movie_title):
    cb = get_content_based_recs(movie_title)
    cf = get_community_recs(movie_title)
    
    hybrid = pd.concat([cb, cf]).drop_duplicates(subset=['id'])
    hybrid['Hybrid_Score'] = ((hybrid['vote_average']/10)*100 * 0.3) + (hybrid.get('CB_Score', 50) * 0.7)
    return hybrid.sort_values('Hybrid_Score', ascending=False).head(20)

# --- HELPER FUNCTION TO RENDER UI CARDS ---
def render_movie_cards(recommendations, score_column):
    html_content = '<div class="scroll-container">'
    
    # EXPLAINABLE AI LOGIC
    if score_column == 'CB_Score':
        match_reason = "🧬 Matches Actors, Director & Plot"
    elif score_column == 'CF_Score':
        match_reason = "⭐ Global Community Rating"
    elif score_column == 'Hybrid_Score':
        match_reason = "✨ AI DNA + Global Rating"
    else:
        match_reason = "🌐 TMDB Search Result"
    
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
        # UNIVERSAL NLP SEARCH: Converts user input (title or plot description) into math
        query_vec = tfidf.transform([search_query])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        best_match_idx = sim_scores.argmax()
        best_score = sim_scores[best_match_idx]
        
        # If the AI finds a logical match in your CSV database
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
            # If the user types gibberish that doesn't match any movie DNA, fail gracefully
            st.info(f"🌐 No local matches. Searching global TMDB database for: **'{search_query}'**")
            topic_results = search_tmdb_topic(search_query)
            
            if topic_results:
                topic_df = pd.DataFrame(topic_results)
                render_movie_cards(topic_df, 'score')
            else:
                st.warning("No movies found for that topic.")
else:
    st.info("👆 Type a movie name (e.g., 'Iron Man') OR a description (e.g., 'race car') to run your AI models!")
