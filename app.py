from flask import Flask, render_template, request
import pickle
import os

from recommender import MovieGenreRecommender 

# ✅ Flask App Starts Here
app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join('model', 'movie_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        genre = request.form['genre']
        if genre:
            try:
                recommendations = model.recommend_by_cluster(genre)
            except Exception as e:
                recommendations = [f"Error: {str(e)}"]
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
