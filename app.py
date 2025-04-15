from flask import Flask, render_template, request
import pickle
import os

# Define the MovieGenreRecommender class (must match the one used during saving)
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

# Initialize Flask app
app = Flask(__name__)

# Load the pickled model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'movie_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Main route
@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        genre = request.form.get('genre', '')
        if genre:
            try:
                recommendations = model.recommend_by_cluster(genre)
            except Exception as e:
                recommendations = [f"Error: {str(e)}"]
    return render_template('index.html', recommendations=recommendations)

# Run the app locally
if __name__ == '__main__':
    app.run(debug=True)
