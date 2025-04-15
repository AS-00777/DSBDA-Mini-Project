import pandas as pd
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load dataset (corrected file name)
df = pd.read_csv('tmdb_5000_movies.csv')

# Parse the genres column from JSON string
def extract_genre_names(genre_str):
    try:
        genres = json.loads(genre_str.replace("'", '"'))  # Fix malformed quotes
        return ' '.join([genre['name'] for genre in genres])
    except:
        return ''

# Create a column with genre names as space-separated strings
df['genre_str'] = df['genres'].apply(extract_genre_names)

# Vectorize genres
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['genre_str'])

# KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Define the model class
class MovieGenreRecommender:
    def __init__(self, df, vectorizer, kmeans):
        self.df = df
        self.vectorizer = vectorizer
        self.kmeans = kmeans

    def recommend_by_genre(self, genre, n=10):
        genre = genre.lower()
        return self.df[self.df['genre_str'].str.lower().str.contains(genre)]['title'].head(n).tolist()

    def recommend_by_cluster(self, genre, n=10):
        test_vec = self.vectorizer.transform([genre])
        cluster = self.kmeans.predict(test_vec)[0]
        return self.df[self.df['cluster'] == cluster]['title'].head(n).tolist()

# Save the trained model
model = MovieGenreRecommender(df, vectorizer, kmeans)

with open('movie_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as movie_model.pkl")
