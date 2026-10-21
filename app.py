
import streamlit as st
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from html import escape


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(circle at 10% 0%, #202044 0%, transparent 35%),
        linear-gradient(135deg, #080812, #11111f 55%, #090910);
    color: #f5f5f7;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

h1, h2, h3, p, label {
    color: #f5f5f7;
}

.hero {
    text-align: center;
    padding: 2rem 1rem 2.5rem;
}

.hero h1 {
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    font-weight: 800;
    letter-spacing: -2px;
    margin-bottom: 0.4rem;
    background: linear-gradient(90deg, #ffffff, #c4b5fd, #f0abfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #aaaac3;
    font-size: 1.1rem;
}

.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 0.3rem;
}

.section-subtitle {
    color: #aaaac3;
    margin-bottom: 1.2rem;
}

.movie-card {
    background: linear-gradient(145deg, #1c1c31, #131320);
    border: 1px solid #30304a;
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 14px;
}

.movie-title {
    font-size: 1rem;
    font-weight: 700;
    margin: 10px 0 8px;
    color: #ffffff;
    overflow-wrap: anywhere;
}

.movie-year {
    font-size: 0.85rem;
    color: #aaaac3;
    margin-bottom: 8px;
}

.movie-overview {
    font-size: 0.9rem;
    color: #c3c3d6;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #191929;
    border-color: #393952;
    border-radius: 10px;
}

div[data-testid="stImage"] img {
    border-radius: 12px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #777790;
    font-size: 0.85rem;
    padding-top: 3rem;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch</h1>
    <p>Discover your next favourite movie.</p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# 4. LOAD MOVIE DATA
# ==================================================

@st.cache_resource
def load_data():
    df = pd.read_pickle("df.pkl")
    tfidf_matrix = pd.read_pickle("tfidf_matrix.pkl")

    if "title" not in df.columns:
        raise ValueError(
            "Your DataFrame must contain a 'title' column."
        )

    if len(df) != tfidf_matrix.shape[0]:
        raise ValueError(
            "The number of movies and TF-IDF rows do not match. "
            "Regenerate both pickle files from the same dataset."
        )

    # Reset index to ensure positions match the TF-IDF rows.
    df = df.reset_index(drop=True)

    return df, tfidf_matrix


try:
    df, tfidf_matrix = load_data()

except Exception as error:
    st.error(f"Unable to load movie data: {error}")
    st.info(
        "Ensure df.pkl and tfidf_matrix.pkl exist in the "
        "same folder as app.py."
    )
    st.stop()


# ==================================================
# 5. TMDB POSTER FUNCTION
# ==================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_poster(movie_title, release_year=None):

    api_key = st.secrets.get("TMDB_API_KEY", "")

    if not api_key:
        return None

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": api_key,
        "query": str(movie_title),
        "include_adult": False
    }

    if release_year:
        try:
            params["year"] = int(release_year)
        except (ValueError, TypeError):
            pass

    try:
        response = requests.get(
            url,
            params=params,
            timeout=8
        )
        response.raise_for_status()

        results = response.json().get("results", [])

        normalized_title = str(movie_title).casefold().strip()

        # Prefer an exact title match and a result with a poster.
        results.sort(
            key=lambda item: (
                item.get("title", "").casefold().strip()
                != normalized_title,
                not bool(item.get("poster_path"))
            )
        )

        for result in results:
            poster_path = result.get("poster_path")

            if poster_path:
                return (
                    "https://image.tmdb.org/t/p/w500"
                    + poster_path
                )

    except (
        requests.RequestException,
        ValueError,
        TypeError,
        KeyError
    ):
        return None

    return None


# ==================================================
# 6. HELPER FUNCTIONS
# ==================================================

def get_release_year(movie):

    if "release_date" in movie.index:
        release_date = str(movie.get("release_date", ""))

        if (
            len(release_date) >= 4
            and release_date[:4].isdigit()
        ):
            return release_date[:4]

    if "year" in movie.index:
        year = str(movie.get("year", ""))

        if len(year) >= 4 and year[:4].isdigit():
            return year[:4]

    return None


def get_movie_overview(movie):

    if "overview" not in movie.index:
        return "No description available."

    overview = movie.get("overview", "")

    if pd.isna(overview) or not str(overview).strip():
        return "No description available."

    return str(overview)


def truncate_text(text, limit=220):

    text = str(text)

    if len(text) <= limit:
        return text

    return text[:limit].rsplit(" ", 1)[0] + "..."


# ==================================================
# 7. RECOMMENDATION ENGINE
# ==================================================

@st.cache_data(show_spinner=False)
def recommend(movie_title, number_of_recommendations=6):

    titles = (
        df["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    normalized_title = movie_title.casefold().strip()

    matching_positions = [
        position
        for position, title in enumerate(titles)
        if title.casefold() == normalized_title
    ]

    if not matching_positions:
        return pd.DataFrame()

    movie_position = matching_positions[0]

    # Calculate similarity against all movies.
    similarity_scores = cosine_similarity(
        tfidf_matrix[movie_position],
        tfidf_matrix
    ).flatten()

    ranked_positions = similarity_scores.argsort()[::-1]

    # Remove the selected movie and duplicate titles.
    recommended_positions = []
    seen_titles = {normalized_title}

    for position in ranked_positions:

        if position == movie_position:
            continue

        title = titles.iloc[position]

        if not title:
            continue

        normalized_recommendation = title.casefold()

        if normalized_recommendation in seen_titles:
            continue

        seen_titles.add(normalized_recommendation)
        recommended_positions.append(position)

        if len(recommended_positions) >= number_of_recommendations:
            break

    return df.iloc[recommended_positions].copy()


# ==================================================
# 8. LIVE MOVIE SEARCH
# ==================================================

st.markdown(
    '<div class="section-title">Discover your next movie</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Type a movie title, choose a match, and discover similar films.'
    '</div>',
    unsafe_allow_html=True
)

movie_titles = (
    df["title"]
    .dropna()
    .astype(str)
    .str.strip()
)

movie_titles = sorted(
    movie_titles[movie_titles != ""].unique().tolist(),
    key=str.casefold
)

if not movie_titles:
    st.error("No movie titles were found in your dataset.")
    st.stop()


# One text field for searching.
search_text = st.text_input(
    "Search movies",
    placeholder="🔎 Start typing a movie name...",
    key="movie_search"
)


# ==================================================
# 9. DISPLAY MATCHES WHILE TYPING
# ==================================================

if search_text.strip():

    matches = [
        title
        for title in movie_titles
        if search_text.casefold().strip() in title.casefold()
    ]

    if matches:

        st.caption(
            f"{len(matches):,} matching movie(s) found. "
            "Choose a title below."
        )

        # Limit the dropdown to avoid an excessively long list.
        matching_options = matches[:50]

        selected_movie = st.selectbox(
            "Matching movies",
            options=matching_options,
            key="movie_match"
        )

        # Recommendations update automatically when the selection changes.
        selected_rows = df[
            df["title"].fillna("").astype(str).str.strip()
            == selected_movie
        ]

        if not selected_rows.empty:

            selected_row = selected_rows.iloc[0]
            selected_year = get_release_year(selected_row)

            st.markdown("---")
            st.markdown(
                '<div class="section-title">Your selected movie</div>',
                unsafe_allow_html=True
            )

            poster_col, details_col = st.columns([1, 3])

            with poster_col:

                with st.spinner("Loading movie poster..."):
                    selected_poster = get_movie_poster(
                        selected_movie,
                        selected_year
                    )

                if selected_poster:
                    st.image(
                        selected_poster,
                        use_container_width=True
                    )
                else:
                    st.info("🎞️ Poster unavailable")

            with details_col:

                st.subheader(selected_movie)

                if selected_year:
                    st.caption(f"📅 Release year: {selected_year}")

                st.write(get_movie_overview(selected_row))

            # ------------------------------------------
            # 10. AUTOMATIC RECOMMENDATIONS
            # ------------------------------------------

            st.markdown("---")
            st.markdown(
                '<div class="section-title">'
                '✨ Movies you might like'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-subtitle">'
                'Recommended based on content similarity.'
                '</div>',
                unsafe_allow_html=True
            )

            number_of_recommendations = st.slider(
                "Number of recommendations",
                min_value=3,
                max_value=12,
                value=6,
                key="recommendation_count"
            )

            with st.spinner("Finding similar movies..."):
                recommendations = recommend(
                    selected_movie,
                    number_of_recommendations
                )

            if recommendations.empty:

                st.warning(
                    "No recommendations found for this movie."
                )

            else:

                columns = st.columns(3)

                for index, (_, movie) in enumerate(
                    recommendations.iterrows()
                ):

                    title = str(
                        movie.get("title", "Unknown title")
                    )

                    year = get_release_year(movie)
                    overview = get_movie_overview(movie)

                    with columns[index % 3]:

                        st.markdown(
                            '<div class="movie-card">',
                            unsafe_allow_html=True
                        )

                        with st.spinner(f"Loading {title}..."):
                            poster_url = get_movie_poster(
                                title,
                                year
                            )

                        if poster_url:
                            st.image(
                                poster_url,
                                use_container_width=True
                            )
                        else:
                            st.info("🎞️ Poster unavailable")

                        # Escape titles before placing them in HTML.
                        safe_title = escape(title)

                        st.markdown(
                            f'<div class="movie-title">'
                            f'{safe_title}</div>',
                            unsafe_allow_html=True
                        )

                        if year:
                            st.markdown(
                                f'<div class="movie-year">'
                                f'📅 {escape(str(year))}</div>',
                                unsafe_allow_html=True
                            )

                        safe_overview = escape(
                            truncate_text(overview)
                        )

                        st.markdown(
                            f'<div class="movie-overview">'
                            f'{safe_overview}</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            '</div>',
                            unsafe_allow_html=True
                        )

    else:

        st.info(
            "No matching movies found. Try another title."
        )

else:

    st.markdown("""
    <div class="movie-card">
        <div class="movie-title">🍿 Ready for movie night?</div>
        <div class="movie-overview">
            Start typing a movie title in the search box above.
            Select a matching movie to see its details and
            discover similar films instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# 11. FOOTER
# ==================================================


st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        padding: 30px 10px 15px;
        margin-top: 50px;
        border-top: 1px solid #343444;
        color: #b8b8cc;
    }
    .footer-name {
        color: #c4b5fd;
        font-size: 20px;
        font-weight: 600;
    }
    </style>

    <div class="footer">
        <p>Discover your next favourite movie.</p>
        <p>Developed with ❤️ by</p>
        <div class="footer-name">Mahi Sharma</div>
        <p>© 2026 CineMatch</p>
    </div>
    """,
    unsafe_allow_html=True
)

