import streamlit as st
import pandas as pd
import numpy as np

# --- 1. PAGE CONFIGURATION ---
# Setting layout to 'wide' uses the whole screen, making it look like a real website
st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🍿", layout="wide")

# --- 2. CUSTOM CSS FOR A BEAUTIFUL UI ---
# This injects HTML/CSS to make the text and cards look premium
st.markdown("""
    <style>
    .main-title {
        font-size: 3.5rem;
        color: #E50914; /* Netflix Red */
        text-align: center;
        font-weight: 900;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #b3b3b3;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .movie-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        height: 100%;
        border: 1px solid #333;
    }
    .match-score {
        color: #46d369; /* Netflix Green for match percentage */
        font-weight: bold;
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. DATA LOADING & CACHING ---
@st.cache_data
def load_and_prep_data():
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('ml-100k/u.data', sep='\t', names=column_names)
    
    movie_cols = ['item_id', 'title']
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=movie_cols)
    
    merged_df = pd.merge(df, movies, on='item_id')
    movie_matrix = merged_df.pivot_table(index='user_id', columns='title', values='rating')
    
    ratings_count = pd.DataFrame(merged_df.groupby('title')['rating'].count())
    ratings_count.rename(columns={'rating': 'num_of_ratings'}, inplace=True)
    
    return movie_matrix, ratings_count, movies['title'].sort_values().unique()

movie_matrix, ratings_count, movie_list = load_and_prep_data()

# --- 4. RECOMMENDER LOGIC ---
def get_recommendations(movie_name, min_reviews=50):
    user_movie_rating = movie_matrix[movie_name]
    similar_movies = movie_matrix.corrwith(user_movie_rating)
    
    corr_movie = pd.DataFrame(similar_movies, columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings_count['num_of_ratings'])
    
    recommendations = corr_movie[corr_movie['num_of_ratings'] > min_reviews].sort_values('Correlation', ascending=False)
    return recommendations.iloc[1:6]

# --- 5. UI LAYOUT ---
# Hero Section
st.markdown('<p class="main-title">🍿 CineMatch</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Your Personalized Movie Recommender System</p>', unsafe_allow_html=True)

# Create 3 Tabs for the 3 Group Members
tab1, tab2, tab3 = st.tabs([
    "👥 Collaborative Filtering", 
    "📖 Content-Based Filtering", 
    "🤖 Hybrid Model"
])

# --- TAB 1: COLLABORATIVE FILTERING ---
with tab1:
    st.markdown("### Item-Based Collaborative Filtering")
    st.write("Find movies similar to your favorites based on community rating patterns.")
    
    # Put the search bar and button side-by-side for a cleaner look
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        selected_movie = st.selectbox("Search for a movie you love:", movie_list, key="cf_select", label_visibility="collapsed")
    with col_btn:
        generate_btn = st.button("Recommend", key="cf_btn", use_container_width=True, type="primary")

    if generate_btn:
        with st.spinner('Crunching the numbers...'):
            try:
                recs = get_recommendations(selected_movie)
                st.success("Top 5 Matches Found!")
                
                # Create 5 columns for a horizontal movie display
                cols = st.columns(5)
                for i, (index, row) in enumerate(recs.iterrows()):
                    with cols[i]:
                        # HTML Card for each movie
                        st.markdown(f'''
                            <div class="movie-card">
                                <h4>🎬 {index}</h4>
                                <p class="match-score">{row['Correlation']*100:.0f}% Match</p>
                                <p style="font-size: 0.8rem; color: #888;">{int(row['num_of_ratings'])} community reviews</p>
                            </div>
                        ''', unsafe_allow_html=True)
                        
            except Exception as e:
                st.error("Not enough data to find correlations for this specific movie. Try a more popular title like 'Star Wars' or 'Toy Story'.")

# --- TAB 2: CONTENT-BASED FILTERING ---
with tab2:
    st.markdown("### Content-Based Filtering")
    st.info("🛠️ **Under Construction:** Member 2 will build this section! This model will recommend movies based on genres, directors, and descriptions.")

# --- TAB 3: HYBRID MODEL ---
with tab3:
    st.markdown("### Hybrid Model")
    st.info("🛠️ **Under Construction:** Member 3 will build this section! This model will combine the power of Collaborative and Content-Based filtering for ultimate accuracy.")
