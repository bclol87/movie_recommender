# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CineMatch Pro", page_icon="🍿", layout="wide")

# --- PRIME VIDEO INSPIRED CSS ---
st.markdown("""
    <style>
    /* Deep Dark Background */
    .stApp {
        background-color: #0f171e !important;
        color: #ffffff !important;
    }
    
    /* Global Text Colors */
    h1, h2, h3, p, span, div { color: #ffffff !important; }

    /* Netflix/Prime Style Typography */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #ffffff, #8197a4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .main-title-orange { color: #00A8E1 !important; -webkit-text-fill-color: #00A8E1 !important; } /* Prime Blue */

    .sub-title {
        font-size: 1rem;
        color: #8197a4 !important;
        margin-bottom: 2rem;
    }

    /* Category Header with Prime Blue accent */
    .category-header {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-left: 0px;
        border-left: 4px solid #00A8E1;
        padding-left: 15px;
    }

    /* Horizontal Scroll Container */
    .scroll-container {
        display: flex;
        overflow-x: auto;
        gap: 15px;
        padding: 20px 0;
        scrollbar-width: none; /* Hide scrollbar for Firefox */
    }
    .scroll-container::-webkit-scrollbar { display: none; } /* Hide scrollbar for Chrome/Safari */

    /* Cinematic Movie Cards */
    .movie-card {
        flex: 0 0 220px;
        background: #1a242f;
        border-radius: 12px;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
        position: relative;
        border: 1px solid rgba(255,255,255,0.05);
    }

    /* Prime Hover Effect: Scale and Glow */
    .movie-card:hover {
        transform: scale(1.08);
        z-index: 10;
        box-shadow: 0 15px 30px rgba(0,0,0,0.5);
        border-color: #00A8E1;
    }

    .movie-poster {
        width: 100%;
        height: 320px;
        object-fit: cover;
    }

    .card-content {
        padding: 15px;
        background: linear-gradient(to top, #1a242f 80%, transparent);
    }

    .movie-title {
        font-size: 1rem;
        font-weight: 700;
        height: 2.5rem;
        overflow: hidden;
        margin-bottom: 5px;
    }

    .match-score {
        color: #46d369 !important; /* Green like Netflix/Prime percentage */
        font-weight: 700;
        font-size: 0.9rem;
    }

    /* Modern Glassmorphism Details Button */
    .watch-btn {
        display: block;
        text-align: center;
        background: rgba(255, 255, 255, 0.1);
        color: white !important;
        text-decoration: none;
        padding: 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 10px;
        transition: 0.3s;
    }
    .watch-btn:hover {
        background: #00A8E1;
        color: white !important;
    }

    /* Input Styling */
    .stTextInput input {
        background-color: #1a242f !important;
        color: white !important;
        border: 1px solid #303b44 !important;
    }
    </style>
""", unsafe_allow_html=True)

def render_movie_cards(recommendations, score_column):
    html_content = '<div class="scroll-container">'
    
    for i, (_, row) in enumerate(recommendations.iterrows()):
        poster_url, overview, movie_link = fetch_movie_details(row['title'])
        score = row.get(score_column, 85)
        
        html_content += f"""
        <div class="movie-card">
            <img src="{poster_url}" class="movie-poster">
            <div class="card-content">
                <div class="movie-title">{row['title']}</div>
                <div class="match-score">{score:.0f}% Match</div>
                <div style="font-size: 0.7rem; color: #8197a4; margin-top: 5px;">AI Curated</div>
                <a href="{movie_link}" target="_blank" class="watch-btn">DETAILS</a>
            </div>
        </div>
        """
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)
