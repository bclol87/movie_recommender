import streamlit as st
import pandas as pd

from movie_logic import (
movies, tfidf, tfidf_matrix,
fetch_movie_details,
search_tmdb_topic,
get_content_based_recs,
get_community_recs,
get_hybrid_recs
)

from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CineMatch Pro", layout="wide")

# ------------------------------------------------

# NETFLIX STYLE CSS

# ------------------------------------------------

st.markdown("""

<style>

.stApp{
background-color:#0f0f0f;
color:white;
}

.main-title{
font-size:48px;
font-weight:900;
text-align:center;
color:#E50914;
margin-bottom:20px;
}

/* HERO BANNER */

.hero{
position:relative;
height:420px;
border-radius:15px;
overflow:hidden;
margin-bottom:40px;
}

.hero img{
width:100%;
height:100%;
object-fit:cover;
filter:brightness(60%);
}

.hero-content{
position:absolute;
left:40px;
bottom:60px;
}

.hero-title{
font-size:46px;
font-weight:800;
}

.hero-desc{
width:420px;
font-size:14px;
color:#ddd;
margin-top:10px;
}

.watch-btn{
background:#E50914;
padding:10px 20px;
border-radius:8px;
font-weight:bold;
text-decoration:none;
color:white;
}

/* ROW TITLE */

.category-header{
font-size:22px;
font-weight:700;
margin-top:20px;
margin-bottom:10px;
}

/* HORIZONTAL SLIDER */

.scroll-container{
display:flex;
overflow-x:auto;
gap:20px;
padding:10px;
}

.scroll-container::-webkit-scrollbar{
display:none;
}

/* MOVIE CARD */

.movie-card{
min-width:180px;
transition:0.25s;
cursor:pointer;
}

.movie-card:hover{
transform:scale(1.1);
}

.movie-card img{
width:100%;
border-radius:8px;
}

.movie-title{
font-size:14px;
font-weight:600;
margin-top:6px;
}

.movie-meta{
font-size:12px;
color:#aaa;
}

</style>

""", unsafe_allow_html=True)

# ------------------------------------------------

# HEADER

# ------------------------------------------------

st.markdown(
"""

<div class="main-title">
CineMatch <span style="color:white;">Pro</span>
</div>
""",
unsafe_allow_html=True
)

# ------------------------------------------------

# HERO BANNER

# ------------------------------------------------

poster, overview, link = fetch_movie_details("Lost in Space")

st.markdown(f"""

<div class="hero">

<img src="{poster}">

<div class="hero-content">

<div class="hero-title">
Lost in Space
</div>

<div class="hero-desc">
{overview}
</div>

<br>

<a class="watch-btn" href="{link}" target="_blank">
▶ Watch
</a>

</div>

</div>
""", unsafe_allow_html=True)

# ------------------------------------------------

# SEARCH BAR

# ------------------------------------------------

search_query = st.text_input(
"",
placeholder="Search movie title or description..."
)

st.divider()

# ------------------------------------------------

# RENDER MOVIE CARDS

# ------------------------------------------------

def render_movie_cards(recommendations, score_column):

```
html = '<div class="scroll-container">'

for _, row in recommendations.iterrows():

    poster, overview, link = fetch_movie_details(row['title'])

    score = row.get(score_column, 80)

    html += f"""

    <div class="movie-card">

    <img src="{poster}">

    <div class="movie-title">{row['title']}</div>

    <div class="movie-meta">
    ⭐ {score:.0f}% Match
    </div>

    </div>

    """

html += "</div>"

st.markdown(html, unsafe_allow_html=True)
```

# ------------------------------------------------

# SEARCH LOGIC

# ------------------------------------------------

if search_query:

```
with st.spinner("AI recommending movies..."):

    query_vec = tfidf.transform([search_query])

    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    best_match_idx = sim_scores.argmax()

    best_score = sim_scores[best_match_idx]

    if best_score > 0:

        selected_movie = movies.iloc[best_match_idx]['title']

        st.success(f"AI matched movie: {selected_movie}")

        st.markdown('<div class="category-header">✨ Hybrid Top Picks</div>', unsafe_allow_html=True)
        render_movie_cards(get_hybrid_recs(selected_movie),'Hybrid_Score')

        st.markdown('<div class="category-header">👥 Community Favorites</div>', unsafe_allow_html=True)
        render_movie_cards(get_community_recs(selected_movie),'CF_Score')

        st.markdown('<div class="category-header">🎭 AI Similar Movies</div>', unsafe_allow_html=True)
        render_movie_cards(get_content_based_recs(selected_movie),'CB_Score')

    else:

        st.info("Searching global movie database...")

        topic_results = search_tmdb_topic(search_query)

        if topic_results:

            topic_df = pd.DataFrame(topic_results)

            st.markdown('<div class="category-header">🌍 Global Results</div>', unsafe_allow_html=True)

            render_movie_cards(topic_df,'score')

        else:

            st.warning("No movies found.")
```

else:

```
st.info("Search a movie or description to get AI recommendations.")
```
