import sys
import os

# Ajouter le chemin du dossier parent dans sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from main import app

def identity(x):
    return x

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict(client):
    with app.app_context():
        data_1 = {
            "text": ["Ceci est une phrase de test."]
        }

        data_2 = {
            "text": ["Ceci est une phrase de test au sujet de Python."]
        }
        # response = predict(data_1)
        response = client.post('/predict', json=data_1)
        # Récupérer le contenu JSON de la réponse 1
        response_json = response.get_json()

        assert response_json == {"tags": []}

        response = client.post('/predict', json=data_2)
    
        # Récupérer le contenu JSON de la réponse 2 
        response_json = response.get_json()

        assert response_json == {"tags": ["python"]}