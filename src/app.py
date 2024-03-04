from pickle import load
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Cargar el modelo, el vectorizador y la data desde el archivo .sav

knn_model = load(open(r"/workspace/flask-deploy-ml-excercise/src/knn_neighbors-6_algorithm-brute_metric-cosine.sav", "rb"))

total_data = load(open(r"/workspace/flask-deploy-ml-excercise/src/total_data.sav", "rb"))
    
vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b', lowercase=True)
vectorizer.fit(total_data['tags'])
    
def recommend_similar_movies(movie_title):
    input_vector = vectorizer.transform([movie_title])
    distances, indices = knn_model.kneighbors(input_vector)
    recommended_movies = [(total_data["title"][i], distances[0][j]) for j, i in enumerate(indices[0])]
    return recommended_movies

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    movie_title = request.form['movie-title']
    recommended_movies = recommend_similar_movies(movie_title)
    return render_template('recommendations.html', movie_title=movie_title, recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)