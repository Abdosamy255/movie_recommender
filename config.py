"""
Configuration module for CineMatch Movie Recommender System.

This module contains all constants, API settings, and configuration
parameters used throughout the application.
"""

import os
from typing import Final

# ─────────────────────────────────────────────────────────────────────────────
# API Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_KEY: Final[str] = None  # Will be loaded from secrets in app
BASE_URL: Final[str] = "https://api.themoviedb.org/3/movie/"
IMAGE_BASE_URL: Final[str] = "https://image.tmdb.org/t/p/w500/"
PLACEHOLDER_IMAGE: Final[str] = "https://via.placeholder.com/500x750/100e1a/c084fc?text=No+Image"
API_TIMEOUT: Final[int] = 6

# ─────────────────────────────────────────────────────────────────────────────
# Data Files
# ─────────────────────────────────────────────────────────────────────────────

MOVIE_LIST_FILE: Final[str] = "movie_list.pkl"
SIMILARITY_FILE: Final[str] = "similarity.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# Recommendation Settings
# ─────────────────────────────────────────────────────────────────────────────

NUM_RECOMMENDATIONS: Final[int] = 5
CACHE_TTL: Final[int] = 86_400  # 24 hours in seconds

# ─────────────────────────────────────────────────────────────────────────────
# UI Configuration
# ─────────────────────────────────────────────────────────────────────────────

APP_TITLE: Final[str] = "CineMatch · Movie Recommender"
APP_ICON: Final[str] = "🎬"
PAGE_LAYOUT: Final[str] = "wide"
SIDEBAR_STATE: Final[str] = "collapsed"

# ─────────────────────────────────────────────────────────────────────────────
# Error Messages
# ─────────────────────────────────────────────────────────────────────────────

ERROR_MISSING_API_KEY: Final[str] = (
    "**`TMDB_API_KEY` not found.** "
    "Add it to `.streamlit/secrets.toml` and restart the app."
)
ERROR_MISSING_DATA_FILE: Final[str] = (
    "Missing data file: `{file}`. "
    "Please ensure both `movie_list.pkl` and `similarity.pkl` are in the app directory."
)
ERROR_CORRUPT_DATA_FILE: Final[str] = (
    "Failed to load data file: `{file}`. The file may be corrupted or invalid."
)
ERROR_MOVIE_NOT_FOUND: Final[str] = (
    "'{movie}' not found in the dataset. Please try another movie."
)
ERROR_API_FETCH: Final[str] = (
    "Unable to fetch movie details from TMDB API. Using placeholder image."
)
