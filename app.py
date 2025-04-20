from flask import Flask, request, jsonify, render_template
import pickle

with open('model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentiment', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    app.run(debug=True)