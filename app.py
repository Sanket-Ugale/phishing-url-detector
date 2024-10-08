from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your models here
loaded_model = joblib.load('path/to/trained_model.pkl')
loaded_vectorizer = joblib.load('path/to/vectorizer.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        input_url = request.form['url_input']
        new_input_tfidf = loaded_vectorizer.transform([input_url])
        prediction = loaded_model.predict(new_input_tfidf)[0]
        result = "Good" if prediction == 1 else "Bad"
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run()
