from flask import Flask, request, jsonify, current_app
import mlflow
import joblib
import tensorflow_hub as hub
import os
import multiprocessing
from multiprocessing import Manager, Pool
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import re
from preprocessing import identity

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

app = Flask(__name__)

def get_wordnet_pos(tag):
    """
    Convertit un tag de partie du discours (POS) en un format compatible avec WordNetLemmatizer.

    Args:
        tag (str): Tag de partie du discours (POS) fourni par pos_tag.

    Returns:
        wordnet.POS: POS compatible avec WordNetLemmatizer.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_token(token):
    """
    Nettoie un token en supprimant les symboles et les chiffres.

    Args:
        token (str): Le token à nettoyer.

    Returns:
        str: Le token nettoyé.
    """
    token = re.sub(r'[^\w\s]', '', token)  # Supprimer la ponctuation
    token = re.sub(r'\d', '', token)       # Supprimer les chiffres
    return token

def preprocess_text(text):
    """
    Prétraite un texte en effectuant des opérations telles que la substitution de certains termes,
    la tokenisation, le POS tagging, le lemmatisation et la suppression des stopwords.

    Args:
        text (str): Le texte à prétraiter.

    Returns:
        list of str: Liste de tokens lemmatisés et filtrés.
    """
    # Initialiser le lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Liste des langues et technologies à remplacer
    languages_and_technologies = [
        "C#", "C", "C++", "R", "Go", "VB", "F#", "JS", "D", "ML", "J#", "PL/I", "PL/SQL"
    ]

    substitutions = [
        (r"C\+\+", "cplusplus"),
        (r"C\#", "csharp"),
        (r"F\#", "fsharp"),
        (r"J\#", "jsharp"),
        (r"PL\/I", "pldashi"),
        (r"PL\/SQL", "pldashsql")
    ]

    reverse_substitutions = {v: re.sub(r'\\', '', k).lower() for k, v in substitutions}

    for language, remplacement in substitutions:
        text = re.sub(language, remplacement, text)

    # Tokenisation
    words = word_tokenize(text.lower())  # Convertit en minuscules pour une cohérence

    # POS tagging pour garder seulement les noms et les verbes
    tagged_words = pos_tag(words)

    lemmatized_tokens = []
    for token, pos in tagged_words:
        cleaned_token = clean_token(token)
        if cleaned_token:  # Vérifier que le token n'est pas vide après nettoyage
            lemmatized_token = lemmatizer.lemmatize(cleaned_token, get_wordnet_pos(pos))
            lemmatized_tokens.append(lemmatized_token)

    stop_words = set(stopwords.words('english'))
    filtered_lemmatized_tokens = [
        token for token in lemmatized_tokens
        if token not in stop_words
        and (len(token) > 2 or token in languages_and_technologies)
    ]

    filtered_lemmatized_tokens = [reverse_substitutions.get(token, token) for token in filtered_lemmatized_tokens]

    return filtered_lemmatized_tokens

type_vectorizer = "tfidf"

if type_vectorizer == "use":
    # URL du modèle Universal Sentence Encoder
    model_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2"
    vectorizer_model = hub.KerasLayer(model_url)
elif type_vectorizer == "tfidf":
    vectorizer_model = joblib.load('vectorizers/tfidf_vectorizer.pkl')

# Chargement du modèle de prédiction
model = joblib.load(f'models/{type_vectorizer}_model.pkl')

mlb = joblib.load('mlbs/mlb.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    if 'text' not in data:
        return jsonify({'error': 'Missing text key in JSON payload'}), 400
    
    texts = data['text']
    # Preprocessing du texte
    if(type_vectorizer == 'use'):        
        # Avec un embedding Universal Sentence Encoder
        # Si texts est une chaîne de caractères on la convertit en liste de chaîne de caractères
        if isinstance(texts, str):
            texts = [texts]        

        X = vectorizer_model(texts).numpy()
        
        
    elif(type_vectorizer == 'tfidf'):
        # Avec une vectorisation TF-IDF
        if  not isinstance(texts, str):
            text = " ".join(texts)

        preprocessed_text = preprocess_text(text)

        X = vectorizer_model.transform([preprocessed_text]).toarray()

    # Prédiction des tags et inversion de la binarisation pour retrouver les mots réels
    tags = model.predict(X)

    tags = mlb.inverse_transform(tags)    
    tags = [tag for sublist in tags for tag in sublist]

    return jsonify({'tags': tags})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))  # Azure définira la variable PORT
    app.run(host="0.0.0.0", port=port)
