import os
import pickle
from flask import Flask, render_template, request

# Define MovieGenreRecommender class
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

# Flask App Starts Here
app = Flask(__name__)

# Update the model path (you can print the absolute path for debugging)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'movie_model.pkl')
print(f"Loading model from: {MODEL_PATH}")

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: The model file was not found.")
    model = None

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        genre = request.form['genre']
        if genre and model:
            try:
                recommendations = model.recommend_by_cluster(genre)
            except Exception as e:
                recommendations = [f"Error: {str(e)}"]
        else:
            recommendations = ["Model not loaded properly."]
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
