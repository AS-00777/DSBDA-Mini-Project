
import os
from flask import Flask, render_template, request
from recommender import MovieGenreRecommender  # Import the class

app = Flask(__name__)
import pickle
# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'movie_model.pkl')
print(f"Loading model from: {MODEL_PATH}")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

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
