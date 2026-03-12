# --- UPDATED CSS SECTION - Replace your existing CSS with this ---
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: #0a0a0a;
        color: #ffffff;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu, footer, header {
        display: none !important;
    }
    
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Search Bar */
    div[data-testid="stTextInput"] {
        position: fixed;
        top: 30px;
        right: 4%;
        z-index: 9999;
        width: 400px;
    }
    
    div[data-testid="stTextInput"] input {
        background: rgba(20, 20, 20, 0.9) !important;
        border: 2px solid rgba(229, 9, 20, 0.5) !important;
        border-radius: 40px !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 15px 25px !important;
    }
    
    /* Hero Banner */
    .hero-banner {
        height: 85vh;
        position: relative;
        display: flex;
        align-items: center;
        padding: 0 6%;
        background-size: cover !important;
        background-position: center !important;
    }
    
    .hero-vignette {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(77deg, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0.5) 50%, rgba(0,0,0,0.2) 100%);
        pointer-events: none;
    }
    
    .hero-bottom-fade {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 150px;
        background: linear-gradient(to top, #0a0a0a 0%, transparent 100%);
        pointer-events: none;
    }
    
    .hero-content {
        position: relative;
        z-index: 10;
        max-width: 700px;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 900;
        margin-bottom: 20px;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .hero-meta {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.8);
        margin-bottom: 30px;
        line-height: 1.6;
    }
    
    /* Buttons */
    .btn-row {
        display: flex;
        gap: 15px;
    }
    
    .btn-play, .btn-info {
        padding: 12px 30px;
        border-radius: 40px;
        font-weight: 600;
        font-size: 1.1rem;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
        border: none;
    }
    
    .btn-play {
        background: #e50914;
        color: white;
    }
    
    .btn-info {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Section Styles */
    .section-container {
        padding: 0 6%;
        margin-top: -40px;
        position: relative;
        z-index: 20;
    }
    
    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
    }
    
    .section-link {
        color: rgba(255,255,255,0.6);
        text-decoration: none;
        font-size: 0.9rem;
    }
    
    /* Horizontal Scroll Container - FIXED */
    .scroll-container {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        gap: 15px !important;
        padding: 10px 0 30px 0 !important;
        margin: 0 !important;
        width: 100% !important;
        min-height: 180px !important;
        scroll-behavior: smooth !important;
        -webkit-overflow-scrolling: touch !important;
    }
    
    .scroll-container::-webkit-scrollbar {
        display: none !important;
    }
    
    /* Movie Cards - FIXED */
    .movie-card {
        flex: 0 0 280px !important;
        width: 280px !important;
        height: 158px !important;
        border-radius: 4px !important;
        overflow: hidden !important;
        position: relative !important;
        cursor: pointer !important;
        transition: transform 0.3s ease !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        background-color: #1a1a1a !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    .movie-card:hover {
        transform: scale(1.05) !important;
        z-index: 30 !important;
        border-color: #e50914 !important;
    }
    
    .movie-card img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        display: block !important;
    }
    
    /* Card Overlay */
    .card-overlay {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        background: linear-gradient(to top, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0.3) 70%) !important;
        opacity: 0 !important;
        transition: opacity 0.3s ease !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-end !important;
        padding: 15px !important;
        pointer-events: none !important;
    }
    
    .movie-card:hover .card-overlay {
        opacity: 1 !important;
    }
    
    .card-match {
        color: #46d369 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        margin-bottom: 5px !important;
    }
    
    .card-title {
        color: white !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    /* Badges */
    .badge-top10 {
        position: absolute !important;
        top: 10px !important;
        right: 10px !important;
        background: #e50914 !important;
        color: white !important;
        font-size: 0.7rem !important;
        font-weight: 900 !important;
        padding: 3px 6px !important;
        border-radius: 2px !important;
        z-index: 5 !important;
    }
    
    .badge-recent {
        position: absolute !important;
        bottom: 10px !important;
        left: 10px !important;
        background: rgba(0,0,0,0.7) !important;
        color: white !important;
        font-size: 0.7rem !important;
        padding: 3px 8px !important;
        border-radius: 12px !important;
        z-index: 5 !important;
    }
    
    /* Top 10 Cards - FIXED */
    .top10-wrapper {
        flex: 0 0 300px !important;
        width: 300px !important;
        display: flex !important;
        position: relative !important;
        cursor: pointer !important;
        height: 200px !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    .top10-number {
        font-size: 12rem !important;
        font-weight: 900 !important;
        color: transparent !important;
        -webkit-text-stroke: 3px #e50914 !important;
        position: absolute !important;
        left: -30px !important;
        bottom: -40px !important;
        z-index: 1 !important;
        line-height: 1 !important;
    }
    
    .top10-card {
        flex: 1 !important;
        height: 100% !important;
        border-radius: 4px !important;
        overflow: hidden !important;
        position: relative !important;
        z-index: 2 !important;
        margin-left: 50px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        background-color: #1a1a1a !important;
    }
    
    .top10-card img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        display: block !important;
    }
    
    /* Debug info styling */
    .debug-info {
        color: #999 !important;
        font-size: 0.8rem !important;
        margin: 5px 0 !important;
    }
    
    /* Force visibility for all cards */
    .stMarkdown {
        width: 100% !important;
        overflow: visible !important;
    }
    </style>
""", unsafe_allow_html=True)
