# Objectif
Cette API a pour but d'envoyer des mots-clés à partir d'une question écrite par un utilisateur en se servant d'un modèle de machine-learning
# Utilisation de l'API
## Endpoint
```
/predict
```
## Données
Les données doivent être envoyé par la méthode POST, et être sous cette forme :
```
{
    "text": "Ceci est une phrase d'exemple."
}
```
## Retour de l'API
Le retour aura cette forme:
```
{
    "tags": ["Tags 1", "Tags 2"...]
}
```
La liste de mots-clés contiendra un nombre plus ou moins important de propositions et pourra être vide si aucun mot-clé pertinent ne puisse être trouvé pour la question écrite.
# Contenu du projet
README.md
