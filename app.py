
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #100d12, #21121a);
    color: #ffffff;
}
.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}
h1, h2, h3, p, label {
    color: #ffffff !important;
}
.hero {
    background: linear-gradient(120deg, #421525, #21121a, #15121c);
    padding: 35px;
    border-radius: 20px;
    border: 1px solid #53303d;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}
.hero p {
    color: #d7c4cc !important;
    font-size: 17px;
}
.movie-card {
    background: #241b23;
    border: 1px solid #45313d;
    padding: 18px;
    border-radius: 14px;
    min-height: 180px;
    margin-bottom: 15px;
}
.movie-card h3 {
    font-size: 19px;
    margin-top: 8px;
}
.movie-card p {
    font-size: 13px;
    color: #d2c4cc !important;
}
.tag {
    color: #ff829e;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 1px;
}
.stButton > button {
    background: #e54868;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: bold;
    min-height: 42px;
}
.stButton > button:hover {
    background: #c92e51;
    color: white;
    border: none;
}
div[data-baseweb="select"] > div {
    background-color: #241b23;
}
hr {
    border-color: #45313d;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_data():
    with open("df.pkl", "rb") as f:
        df = pickle.load(f)

    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    return df, tfidf_matrix


try:
    df, tfidf_matrix = load_data()
except Exception as e:
    st.error("Could not load your movie files.")
    st.write("Check that df.pkl and tfidf_matrix.pkl are in this folder.")
    st.code(str(e))
    st.stop()

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <div class="tag">YOUR PERSONAL MOVIE DISCOVERY PLATFORM</div>
    <h1>🎬 CineMatch</h1>
    <p>Not sure what to watch? Discover your next favourite movie.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SEARCH SECTION ----------------
st.subheader("🔎 Find a movie")

search = st.text_input(
    "Search",
    placeholder="Type a movie name, e.g. Toy Story..."
)

all_titles = (
    df["title"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .sort_values()
    .tolist()
)

if search.strip():
    filtered_titles = [
        title for title in all_titles
        if search.strip().lower() in title.lower()
    ]
else:
    filtered_titles = all_titles

if not filtered_titles:
    st.warning("No movies found. Try another title.")
    st.stop()

selected_movie = st.selectbox(
    "Choose a movie",
    filtered_titles,
    index=0
)

number_of_movies = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=6
)

# ---------------- RECOMMENDATION LOGIC ----------------
def recommend(movie_title, n=6):
    movie_indices = df.index[
        df["title"].astype(str) == movie_title
    ].tolist()

    if not movie_indices:
        return pd.DataFrame()

    movie_index = movie_indices[0]

    scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Highest similarity first, excluding the selected movie
    similar_indices = scores.argsort()[::-1]
    similar_indices = [
        i for i in similar_indices
        if i != movie_index
    ][:n]

    result = df.iloc[similar_indices].copy()
    result["similarity_score"] = scores[similar_indices]

    return result


# ---------------- DISPLAY MOVIES ----------------
if st.button("✨ Find Similar Movies", use_container_width=True):

    recommendations = recommend(
        selected_movie,
        number_of_movies
    )

    if recommendations.empty:
        st.warning("No recommendations could be generated.")
    else:
        st.divider()

        st.subheader(f"Because you selected: {selected_movie}")
        st.caption("Movies selected using content similarity")

        # Selected movie information
        selected_row = df[
            df["title"].astype(str) == selected_movie
        ].iloc[0]

        overview = str(
            selected_row.get("overview", "")
        ).strip()

        if overview:
            st.info(overview)

        # Recommendation cards
        cols = st.columns(3)

        for i, (_, movie) in enumerate(
            recommendations.iterrows()
        ):
            with cols[i % 3]:
                title = str(movie.get("title", "Unknown"))
                genres = str(movie.get("genres", ""))
                overview = str(movie.get("overview", ""))
                rating = movie.get("vote_average", 0)

                try:
                    rating = float(rating)
                    rating_text = f"{rating:.1f}/10"
                except (ValueError, TypeError):
                    rating_text = "Not rated"

                if genres == "nan" or not genres.strip():
                    genres = "Genre unavailable"

                if overview == "nan":
                    overview = ""

                if len(overview) > 160:
                    overview = overview[:160].rsplit(" ", 1)[0] + "..."

                st.markdown(f"""
                <div class="movie-card">
                    <div class="tag">RECOMMENDED FOR YOU</div>
                    <h3>🎞️ {title}</h3>
                    <p>⭐ Rating: {rating_text}</p>
                    <p>🎭 {genres}</p>
                    <p>{overview or "No description available."}</p>
                </div>
                """, unsafe_allow_html=True)

        st.success("Enjoy discovering your next movie!")

st.divider()

st.markdown(
    "<p style='text-align:center;color:#a99aa3;'>"
    "CineMatch · Discover stories worth watching 🍿"
    "</p>",
    unsafe_allow_html=True
)
