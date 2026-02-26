"""
CineMatch: AI-Powered Movie Recommender System

A Streamlit-based web application that provides personalized movie recommendations
using content-based filtering with cosine similarity. It integrates with the TMDB API
to fetch movie posters and information.

Author: Ahmed Ziada
Dataset: TMDB 5000 Movies & Credits
"""

from typing import Tuple, List
import pickle
import os
import logging
import zipfile
import streamlit as st
import requests
import pandas as pd

import config

# ─────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.PAGE_LAYOUT,
    initial_sidebar_state=config.SIDEBAR_STATE,
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ══════════════════════════════════════════
   KEYFRAMES
══════════════════════════════════════════ */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes cardIn {
    from { opacity: 0; transform: translateY(36px) scale(0.96); }
    to   { opacity: 1; transform: translateY(0)   scale(1); }
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 4px 22px rgba(124,58,237,0.4); }
    50%       { box-shadow: 0 4px 38px rgba(168,85,247,0.65); }
}
@keyframes orb {
    0%   { transform: translate(-50%, -50%) scale(1);    opacity: 0.16; }
    50%  { transform: translate(-50%, -50%) scale(1.18); opacity: 0.26; }
    100% { transform: translate(-50%, -50%) scale(1);    opacity: 0.16; }
}

/* ══════════════════════════════════════════
   BASE
══════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background: #08080f !important;
    color: #e8e4dc;
    font-family: 'DM Sans', sans-serif;
}

/* Soft animated purple orb in background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 50%;
    width: 900px; height: 600px;
    background: radial-gradient(ellipse, rgba(109,40,217,0.2) 0%, transparent 68%);
    transform: translate(-50%, -50%);
    animation: orb 9s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}

.main .block-container {
    padding-top: 0 !important;
    max-width: 1120px !important;
    position: relative;
    z-index: 1;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* ══════════════════════════════════════════
   HERO
══════════════════════════════════════════ */
.hero {
    text-align: center;
    padding: 4.5rem 1rem 2.8rem;
    animation: fadeUp 0.9s cubic-bezier(.22,1,.36,1) both;
}
.hero-eyebrow {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.38em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(3.2rem, 8.5vw, 6rem);
    font-weight: 900;
    line-height: 1;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #f5f0ff 20%, #c084fc 60%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 1rem;
}
.hero-subtitle {
    font-size: 0.92rem;
    color: #6b5f80;
    font-weight: 300;
    letter-spacing: 0.04em;
}
.divider {
    width: 48px; height: 2px;
    background: linear-gradient(90deg, transparent, #7c3aed, transparent);
    margin: 1.6rem auto;
}

/* ══════════════════════════════════════════
   SEARCH
══════════════════════════════════════════ */
.search-wrapper {
    max-width: 580px;
    margin: 0 auto 1.8rem;
    padding: 0 1rem;
    animation: fadeUp 0.9s 0.12s cubic-bezier(.22,1,.36,1) both;
}
.search-label {
    font-size: 0.67rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: #4a3f60;
    margin-bottom: 0.6rem;
    font-weight: 500;
}

[data-testid="stSelectbox"] > div > div {
    background: rgba(124,58,237,0.06) !important;
    border: 1px solid rgba(192,132,252,0.22) !important;
    border-radius: 10px !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
    transition: border-color 0.3s, box-shadow 0.3s;
}
[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(192,132,252,0.55) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
}

/* ══════════════════════════════════════════
   BUTTON
══════════════════════════════════════════ */
[data-testid="stButton"] {
    animation: fadeUp 0.9s 0.22s cubic-bezier(.22,1,.36,1) both;
}
[data-testid="stButton"] > button {
    display: block !important;
    margin: 0 auto !important;
    background: linear-gradient(135deg, #5b21b6 0%, #7c3aed 50%, #a855f7 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #f5f0ff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    padding: 0.82rem 2.8rem !important;
    cursor: pointer !important;
    animation: glowPulse 3s ease-in-out infinite !important;
    transition: transform 0.25s ease, filter 0.25s ease !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    filter: brightness(1.15) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) scale(0.99) !important;
}

/* ══════════════════════════════════════════
   SECTION HEADINGS
══════════════════════════════════════════ */
.section-heading {
    text-align: center;
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8e4dc;
    letter-spacing: -0.01em;
    margin: 3rem 0 0.35rem;
    animation: fadeUp 0.7s cubic-bezier(.22,1,.36,1) both;
}
.section-sub {
    text-align: center;
    font-size: 0.67rem;
    color: #4a3f60;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    margin-bottom: 2rem;
    animation: fadeUp 0.7s 0.06s cubic-bezier(.22,1,.36,1) both;
}

/* ══════════════════════════════════════════
   CARDS
══════════════════════════════════════════ */
.cards-row {
    display: flex;
    gap: 16px;
    justify-content: center;
    padding: 0 1rem 3.5rem;
}

.movie-card {
    flex: 1;
    min-width: 0;
    max-width: 195px;
    border-radius: 14px;
    overflow: hidden;
    background: #100e1a;
    border: 1px solid rgba(192,132,252,0.12);
    position: relative;
    opacity: 0;
    animation: cardIn 0.55s cubic-bezier(.22,1,.36,1) forwards;
    transition: transform 0.3s cubic-bezier(.22,1,.36,1),
                box-shadow 0.3s ease,
                border-color 0.3s ease;
}
.movie-card:nth-child(1) { animation-delay: 0.04s; }
.movie-card:nth-child(2) { animation-delay: 0.11s; }
.movie-card:nth-child(3) { animation-delay: 0.18s; }
.movie-card:nth-child(4) { animation-delay: 0.25s; }
.movie-card:nth-child(5) { animation-delay: 0.32s; }

.movie-card:hover {
    transform: translateY(-7px) scale(1.02);
    border-color: rgba(192,132,252,0.45);
    box-shadow: 0 20px 48px rgba(0,0,0,0.55),
                0 0 0 1px rgba(192,132,252,0.2);
}

/* Perfectly uniform 2:3 poster box */
.poster-wrap {
    width: 100%;
    aspect-ratio: 2 / 3;
    overflow: hidden;
    background: linear-gradient(135deg, #1a1028, #0e0c18);
    position: relative;
    display: block;
}
.poster-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center top;
    display: block;
    transition: transform 0.45s cubic-bezier(.22,1,.36,1);
}
.movie-card:hover .poster-wrap img {
    transform: scale(1.06);
}
/* Bottom fade */
.poster-wrap::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(to top,
        rgba(16,14,26,0.8) 0%,
        transparent 50%);
    pointer-events: none;
}

.movie-card-title {
    padding: 0.7rem 0.85rem 0.85rem;
    font-size: 0.8rem;
    font-weight: 400;
    color: #b09ec4;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 2.7em;
}

.rank-badge {
    position: absolute;
    top: 10px; left: 10px;
    z-index: 2;
    background: rgba(109,40,217,0.78);
    color: #f0e8ff;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 3px 9px;
    border-radius: 20px;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(192,132,252,0.3);
}

/* ══════════════════════════════════════════
   MISC
══════════════════════════════════════════ */
[data-testid="stSpinner"] { color: #a78bfa !important; }
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# API Key Configuration
# ─────────────────────────────────────────────────────────────────────────────

try:
    api_key = st.secrets["TMDB_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(config.ERROR_MISSING_API_KEY)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────────────────────────────────────


import zipfile
import pickle
import streamlit as st
import os

@st.cache_data(show_spinner=False)
def load_data():
    # التأكد من وجود الملفات
    if not os.path.exists("movie_list.pkl") or not os.path.exists("similarity.zip"):
        st.error("Missing data files. Make sure 'movie_list.pkl' and 'similarity.zip' exist.")
        return None, None
        
    try:
        # قراءة الداتا بتاعة الأفلام
        with open("movie_list.pkl", "rb") as f:
            movies = pickle.load(f)
            
        # قراءة مصفوفة التشابه من الملف المضغوط مباشرة
        with zipfile.ZipFile("similarity.zip", "r") as z:
            with z.open("similarity.pkl") as f:
                similarity = pickle.load(f)
                
        return movies, similarity
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return None, None
    except Exception as exc:
        logger.error(f"Unexpected error loading data: {exc}")
        st.error(f"An unexpected error occurred while loading data: {exc}")
        return None, None


@st.cache_data(show_spinner=False, ttl=config.CACHE_TTL)
def fetch_movie_poster(movie_id: int) -> str:
    """
    Fetch movie poster URL from TMDB API.
    
    Args:
        movie_id (int): The TMDB movie ID.
    
    Returns:
        str: URL of the movie poster, or placeholder image if unavailable.
    """
    try:
        response = requests.get(
            f"{config.BASE_URL}{movie_id}",
            params={"api_key": api_key, "language": "en-US"},
            timeout=config.API_TIMEOUT,
        )
        response.raise_for_status()
        poster_path = response.json().get("poster_path")
        
        if poster_path:
            logger.debug(f"Successfully fetched poster for movie ID {movie_id}")
            return f"{config.IMAGE_BASE_URL}{poster_path}"
    
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching poster for movie ID {movie_id}")
    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP error fetching poster for movie ID {movie_id}: {e}")
    except Exception as e:
        logger.warning(f"Error fetching poster for movie ID {movie_id}: {e}")
    
    logger.debug(f"Using placeholder image for movie ID {movie_id}")
    return config.PLACEHOLDER_IMAGE


def get_recommendations(
    movie_title: str, 
    movies_df: pd.DataFrame, 
    similarity_matrix: any
) -> Tuple[List[str], List[str]]:
    """
    Generate movie recommendations based on similarity to a selected movie.
    
    Args:
        movie_title (str): Title of the movie to find recommendations for.
        movies_df (pd.DataFrame): DataFrame containing movie information.
        similarity_matrix (any): Similarity scores matrix.
    
    Returns:
        Tuple[List[str], List[str]]: Tuple of (movie titles, poster URLs) for
            the top N recommendations. Returns empty lists if movie not found.
    """
    try:
        # Find the index of the selected movie
        movie_idx_series = movies_df[movies_df["title"] == movie_title].index
        if len(movie_idx_series) == 0:
            logger.warning(f"Movie not found in dataset: {movie_title}")
            st.warning(config.ERROR_MOVIE_NOT_FOUND.format(movie=movie_title))
            return [], []
        
        movie_idx = movie_idx_series[0]
        logger.info(f"Finding recommendations for: {movie_title}")
        
        # Calculate similarity scores and sort
        similarity_scores = enumerate(similarity_matrix[movie_idx])
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top recommendations (skip the first one as it's the movie itself)
        recommended_titles: List[str] = []
        recommended_posters: List[str] = []
        
        for idx, _ in sorted_scores[1:config.NUM_RECOMMENDATIONS + 1]:
            movie_row = movies_df.iloc[idx]
            recommended_titles.append(movie_row["title"])
            recommended_posters.append(fetch_movie_poster(movie_row["movie_id"]))
        
        logger.info(f"Generated {len(recommended_titles)} recommendations")
        return recommended_titles, recommended_posters
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        st.error(f"An error occurred while generating recommendations: {e}")
        return [], []



# ─────────────────────────────────────────────────────────────────────────────
# Main Application UI
# ─────────────────────────────────────────────────────────────────────────────

def render_hero_section() -> None:
    """Render the hero/header section of the application."""
    st.markdown("""
        <div class="hero">
            <p class="hero-eyebrow">AI · Powered</p>
            <h1 class="hero-title">CineMatch</h1>
            <p class="hero-subtitle">Discover films tailored to your taste</p>
            <div class="divider"></div>
        </div>
    """, unsafe_allow_html=True)


def render_recommendation_cards(
    titles: List[str],
    posters: List[str],
    selected_movie: str
) -> None:
    """
    Render the recommendation cards in a grid layout.
    
    Args:
        titles (List[str]): List of recommended movie titles.
        posters (List[str]): List of poster URLs.
        selected_movie (str): The original movie selected by the user.
    """
    st.markdown('<p class="section-heading">Recommended for You</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="section-sub">Based on &nbsp;·&nbsp; {selected_movie}</p>',
        unsafe_allow_html=True
    )

    # Build HTML for recommendation cards
    cards_html = '<div class="cards-row">'
    for rank, (title, poster) in enumerate(zip(titles, posters), 1):
        cards_html += f"""
            <div class="movie-card">
                <span class="rank-badge">#{rank}</span>
                <div class="poster-wrap">
                    <img src="{poster}" alt="{title}" loading="lazy"/>
                </div>
                <div class="movie-card-title">{title}</div>
            </div>"""
    cards_html += "</div>"
    
    st.markdown(cards_html, unsafe_allow_html=True)


def main() -> None:
    """Main application entry point."""
    # Render hero section
    render_hero_section()
    
    # Load data
    logger.info("Loading application data...")
    movies, similarity = load_data()
    
    # Check if data loaded successfully
    if movies is None or similarity is None:
        logger.error("Failed to load required data files")
        return
    
    # Render search interface
    st.markdown('<div class="search-wrapper">', unsafe_allow_html=True)
    st.markdown('<p class="search-label">Choose a movie</p>', unsafe_allow_html=True)
    
    selected_movie = st.selectbox(
        label="Select a movie",
        options=sorted(movies["title"].values),
        label_visibility="collapsed",
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Render recommendation button
    col_btn = st.columns([1, 2, 1])[1]
    with col_btn:
        find_btn = st.button("✦  Find Recommendations")
    
    # Handle recommendation request
    if find_btn:
        logger.info(f"User requested recommendations for: {selected_movie}")
        with st.spinner("Curating your watchlist…"):
            recommended_titles, recommended_posters = get_recommendations(
                selected_movie, 
                movies, 
                similarity
            )
        
        if recommended_titles:
            render_recommendation_cards(
                recommended_titles,
                recommended_posters,
                selected_movie
            )
        else:
            logger.warning(f"No recommendations generated for: {selected_movie}")


if __name__ == "__main__":
    main()