# 🎬 CineMatch — Movie Recommendation System

Discover your next favourite movie with **CineMatch**, a content-based movie recommendation system built using Python and machine learning. It recommends movies similar to the one you select, helping you explore films that match your interests.

## ✨ Features

- 🎥 **Movie Recommendations** — Get recommendations based on your selected movie.
- 🧠 **Content-Based Filtering** — Uses movie metadata and text similarity to find related movies.
- 🔍 **Similarity Matching** — Uses TF-IDF vectorization and cosine similarity to compare movies.
- 🖼️ **Movie Posters** — Fetches movie poster images using The Movie Database (TMDB) API.
- 🌐 **Interactive Web App** — Explore recommendations through a Streamlit interface.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Pandas | Data processing and manipulation |
| Scikit-learn | TF-IDF vectorization and cosine similarity |
| Pickle | Loading preprocessed movie data and similarity features |
| TMDB API | Fetching movie poster images |

## ⚙️ How It Works

1. **Data Preparation:** Movie information is loaded from a preprocessed dataset.
2. **Feature Extraction:** TF-IDF converts textual movie features into numerical vectors.
3. **Similarity Calculation:** Cosine similarity measures how closely movies match.
4. **Recommendation:** The system identifies and displays movies most similar to the selected movie.
5. **Poster Retrieval:** The app uses TMDB to retrieve movie posters.

## 📁 Project Structure

```text
Movie_Recommendations/
│
├── app.py                 # Main Streamlit application
├── df.pkl                 # Preprocessed movie dataset
├── tfidf_matrix.pkl       # TF-IDF feature matrix
├── requirements.txt       # Python dependencies
├── .gitignore             # Files excluded from Git
└── README.md              # Project documentation
```


## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mahisharma011/Movie_Recommendations.git
cd Movie_Recommendations
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the TMDB API

To display movie posters, obtain an API key from [The Movie Database](https://www.themoviedb.org/).

If your application uses Streamlit secrets, create `.streamlit/secrets.toml` locally:

```toml
TMDB_API_KEY = "your_api_key_here"
```

Replace the placeholder with your own key. **Never commit API keys or secrets to GitHub.**

If your code reads the key from a different variable or configuration location, follow the implementation in `app.py`.

### 5. Run the Application

```bash
streamlit run app.py
```

Streamlit will provide a local URL where you can open the application in your browser.

## 🎬 Try It Yourself

Want to see CineMatch in action?

**Start by searching for `Avengers`!** 🦸

Run the app, enter *Avengers* in the movie search box, and explore recommendations for similar movies. Then try your own favourites and discover what CineMatch suggests next!

**Your mission:** Search. Discover. Repeat. 🍿


## 📊 Dataset

The project uses preprocessed movie data and a saved TF-IDF matrix to generate recommendations. The files `df.pkl` and `tfidf_matrix.pkl` must be available in the expected project directory.


## 🎯 Learning Outcomes

Through this project, I explored:

- Content-based recommendation systems.
- Text feature extraction using TF-IDF.
- Cosine similarity for measuring similarity between movies.
- Data processing with Pandas.
- Building interactive applications using Streamlit.
- Managing Python dependencies and Git version control.

## 👩‍💻 Author

**Mahi Sharma**

Second-Year Engineering Student | Artificial Intelligence & Machine Learning Engineer

🔗 **GitHub:** [mahisharma011](https://github.com/mahisharma011)


⭐ If you find this project interesting, consider giving the repository a star!
