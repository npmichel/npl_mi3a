TP6 : Réseaux de neurones à mémoire longue (LSTM)

Guide d'utilisation


Fichiers du projet

Scripts principaux :

1. `lstm_nlp.py` : Implémentation complète du modèle LSTM
   - Fonctions : tokenization, data_preparation, model, fit_model
   - Peut être exécuté de manière autonome pour tester le LSTM

2. `rnn_nlp.py` : Implémentation complète du modèle RNN classique (SimpleRNN)
   - Fonctions : tokenization, data_preparation, model, fit_model
   - Peut être exécuté de manière autonome pour tester le RNN

3. `compare_lstm_rnn.py` : Script de comparaison directe LSTM vs RNN
   - Exécute les deux modèles sur le même corpus
   - Affiche les prédictions côte à côte

4. `analyse_comparaison.py` : Analyse statistique approfondie
   - Effectue 10 entraînements indépendants pour chaque modèle
   - Calcule la stabilité et la confiance moyenne
   - Génère des statistiques détaillées

5. `test_final_tp6.py` : Test final avec analyse complète
   - Un seul entraînement avec résultats détaillés
   - Affiche le top 3 des prédictions
   - Calcule l'entropie de la distribution
   - Format adapté pour inclusion dans le rapport

6. `w2v_pret.py` : Prétraitement des données textuelles
   - Tokenization et normalisation
   - Utilisé par tous les autres scripts


Documents :

1. `RAPPORT_TP6_LSTM.md` : Rapport complet du TP
   - Analyse théorique et expérimentale
   - Résultats détaillés
   - Interprétation des différences LSTM vs RNN

2. `TP6_LSTM_IA_NLP.pdf` : Énoncé du TP


Exécution des scripts

Prérequis :

```bash
# Installation des dépendances avec uv
uv sync
```

Ou si vous utilisez pip :

```bash
pip install tensorflow numpy nltk
```

Exécution :

1. Test du LSTM seul :
```bash
uv run python lstm_nlp.py
```

2. Test du RNN seul :
```bash
uv run python rnn_nlp.py
```

3. Comparaison directe LSTM vs RNN :
```bash
uv run python compare_lstm_rnn.py
```

4. Analyse statistique complète (10 entraînements) :
```bash
uv run python analyse_comparaison.py
```

5. Test final pour le rapport :
```bash
uv run python test_final_tp6.py
```


Résultats attendus

LSTM :
- Meilleure capacité à capturer les dépendances longues
- Plus grande variabilité dans les prédictions (exploration de l'espace)
- Présence du mot syntaxiquement correct dans le top 3
- Confiance plus stable entre les entraînements

RNN :
- Convergence systématique vers une seule prédiction
- Perte de l'information sur le sujet initial
- Confiance élevée mais moins stable
- Incapacité à gérer les dépendances longues (> 10 mots)


Corpus utilisé

Le corpus contient 4 phrases avec des propositions relatives imbriquées :

1. "le chat que le chien que le voisin que l enfant a vu nourrit dort"
2. "le chien que le chat que le voisin que l enfant observe nourrit aboie"
3. "le chat que le chien que le facteur que le voisin connait effraie se cache"
4. "le chien que le chat que le voisin que le facteur aide observe se nourrit"

Ces phrases testent la capacité des modèles à conserver en mémoire le sujet initial ("le chat" ou "le chien") jusqu'au verbe final, malgré de nombreux mots intermédiaires.


Question du TP

"Expliquez pourquoi le RNN échoue ou produit des prédictions instables lorsque la phrase devient longue, alors que le LSTM parvient à prédire un mot plus cohérent."

Réponse synthétique :

Le RNN classique souffre de deux problèmes fondamentaux :

1. Problème du gradient qui s'évanouit : Lors de la rétropropagation à travers 13 étapes temporelles, le gradient diminue exponentiellement, rendant difficile l'apprentissage des dépendances lointaines.

2. Capacité mémoire limitée : Le RNN encode toute l'information dans un unique vecteur d'état caché qui est régulièrement écrasé par les nouvelles informations, causant une perte du contexte initial.

Le LSTM résout ces problèmes grâce à :
- Une cellule mémoire distincte de l'état caché
- Des portes de contrôle (forget, input, output) qui gèrent explicitement le flux d'information
- Un chemin de gradient privilégié qui évite les multiplications répétées

Résultat : Le LSTM maintient l'information sur le sujet initial et place le mot syntaxiquement correct dans son top 3, tandis que le RNN converge systématiquement vers le mot le plus fréquent, ignorant la structure syntaxique.


Auteur

TP réalisé dans le cadre du Master Intelligence Artificielle
Module : Traitement du Langage Naturel
