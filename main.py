from flask import Flask, request, jsonify
# import joblib
import tensorflow_hub as hub
import os

# URL du modèle Universal Sentence Encoder
model_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2"
embed_model = hub.KerasLayer(model_url)

# mlb = joblib.load('mlbs/mlb.pkl')

app = Flask(__name__)

# # Chargement du modèle de prédiction
# model = joblib.load('models/model.pkl')
    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Missing text key in JSON payload'}), 400
    texts = data['text']

    # Si texts est une chaîne de caractères on la convertit en liste de chaîne de caractères
    if isinstance(texts, str):
        texts = [texts] 
    
    # embedding = embed_model(texts).numpy()
    
    # tags = model.predict(embedding)
    # tags = mlb.inverse_transform(tags)
    
    return jsonify({'tags': texts})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))  # Azure définira la variable PORT
    app.run(host="0.0.0.0", port=port)
