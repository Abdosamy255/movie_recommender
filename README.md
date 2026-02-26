# CineMatch: AI-Powered Movie Recommender System

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://github.com/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Discover movies tailored to your taste using AI-powered content-based filtering**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Technical Details](#-technical-details) • [Dataset](#-dataset)

</div>

---

## 🎬 Overview

CineMatch is an intelligent movie recommendation system that leverages **content-based filtering** and **cosine similarity** to suggest movies based on your preferences. Built with Streamlit and powered by the TMDB API, it provides a seamless, modern user interface for discovering your next favorite film.

---

## ✨ Features

- **AI-Powered Recommendations**: Uses cosine similarity algorithm
- **Real-time Movie Data**: Integrates with TMDB API
- **Beautiful Modern UI**: Responsive interface with animations
- **Fast Caching**: Efficient data handling
- **Comprehensive Logging**: Built-in logging support
- **Type Hints**: Full type annotations
- **Error Handling**: Robust error management

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- TMDB API key (free at [themoviedb.org](https://www.themoviedb.org/settings/api))

### Quick Start

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # macOS/Linux
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key**:
   Create `.streamlit/secrets.toml`:
   ```toml
   TMDB_API_KEY = "your_api_key_here"
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

---

## 🚀 Usage

1. Select a movie from the dropdown
2. Click "Find Recommendations"
3. Browse the recommended movies with their posters

---

## 📊 Technical Details

### Algorithm
- **Content-based filtering** with cosine similarity
- TF-IDF feature engineering
- Top 5 recommendations per query

### Technology Stack
- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **Data**: Pandas, NumPy, Scikit-Learn
- **API**: TMDB (requests)

### Project Structure
```
movie-recommender-system-tmdb-dataset-main/
├── app.py                      # Main Streamlit app
├── config.py                   # Configuration
├── requirements.txt            # Dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Documentation
├── movie_list.pkl              # Movie data
├── similarity.pkl              # Similarity matrix
└── .streamlit/
    ├── config.toml             # Streamlit config
    └── secrets.toml            # API keys
```

---

## 📈 Dataset

Trained on **TMDB 5000 Movies Dataset**:
- 4,963 movies
- Genres, keywords, cast, plot summaries
- Source: [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## 🔐 Security

- Store API keys in `.streamlit/secrets.toml`
- Add to `.gitignore` (never commit keys)
- No data storage or tracking

---

## 🐛 Troubleshooting

**TMDB_API_KEY not found?**
- Create `.streamlit/secrets.toml` with your API key

**Missing data files?**
- Ensure `movie_list.pkl` and `similarity.pkl` exist

**Slow recommendations?**
- Normal on first run; results are cached afterward

---

## 📄 License

MIT License

---

## 👤 Author

Abdelrhman samy

---

## 🙏 Acknowledgments

- TMDB for the API and dataset
- Streamlit for the framework
- Scikit-learn for ML utilities

---

<div align="center">

**Made with ❤️ for movie enthusiasts**

</div>
