from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

# URL du modèle Universal Sentence Encoder
model_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2"
embed_model = hub.KerasLayer(model_url)

mlb = joblib.load('mlbs/mlb.pkl')

app = Flask(__name__)

# Chargement du modèle de prédiction
model = joblib.load('models/model.pkl')
    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Missing text key in JSON payload'}), 400
    texts = data['text']
    
    embedding = embed_model(texts).numpy()
    
    tags = model.predict(embedding)
    tags = mlb.inverse_transform(tags)
    
    return jsonify({'tags': tags[0]})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
