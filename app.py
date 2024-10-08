from flask import Flask, render_template, request
import joblib

# Load the trained model and vectorizer
loaded_model = joblib.load('trained_model.pkl')
loaded_vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        # Get the input URL from the form
        input_url = request.form['url_input']
        
        # Transform the input URL using the vectorizer
        input_tfidf = loaded_vectorizer.transform([input_url])
        
        # Predict the result using the model
        prediction = loaded_model.predict(input_tfidf)[0]
        
        # Prepare the result
        result = 'Good' if prediction == 1 else 'Bad'
        
        # Render the result on the same page
        return render_template('index.html', input_url=input_url, result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
