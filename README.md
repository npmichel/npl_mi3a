# TP1 - Prétraitement NLP en Français

## 📝 Description

Ce projet implémente un ensemble complet de fonctions de prétraitement de texte en français dans le cadre du TP1 d'Intelligence Artificielle - Traitement du Langage Naturel (NLP).

**Corpus de test**: Texte sur la Coupe d'Afrique des Nations (CAN) 2025 au Maroc

## 🎯 Objectifs du TP

Le TP couvre les questions suivantes:

1. **Question 1**: Normalisation de texte
2. **Question 2**: Tokenisation avec NLTK
3. **Question 2 bis**: Tokenisation avec spaCy et comparaison
4. **Question 3**: Suppression des stopwords avec NLTK
5. **Question 3 bis**: Suppression des stopwords avec spaCy
6. **Question 4**: Lemmatisation avec spaCy
7. **Question 5**: Calcul des fréquences des mots
8. **Question 6**: Reconnaissance d'entités nommées (NER)
9. **Question 8**: Pipeline de traitement complet
10. **Question 9**: Suppression de la ponctuation
11. **Question 10**: Comparaison Stemming vs Lemmatisation
12. **Question 11**: Extraction de n-grams (bi-grams, tri-grams)
13. **Question 12**: Calcul de la longueur moyenne des mots
14. **Question 13**: Analyse de la taille du vocabulaire
15. **Question 14**: Analyse des fréquences absolues vs relatives

## 🛠️ Installation

### Prérequis

- Python 3.11 ou supérieur
- uv (gestionnaire de paquets)

### Installation des dépendances

```bash
# Installer les dépendances
uv sync

# Télécharger le modèle spaCy français (si pas déjà fait)
uv add https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.8.0/fr_core_news_sm-3.8.0-py3-none-any.whl
```

## 🚀 Utilisation

### Méthode 1: Exécuter le fichier principal (recommandé pour le rendu)

Le fichier `pre_processing.py` contient toutes les fonctions et un bloc de démonstration complet:

```bash
python pre_processing.py
```

Cette commande exécutera tous les tests et affichera:
- Les résultats de chaque fonction
- Un résumé statistique final
- La validation que toutes les fonctions fonctionnent correctement

### Méthode 2: Utiliser le fichier de démonstration

```bash
python demo.py
```

### Méthode 3: Importer les fonctions dans votre propre code

```python
import pre_processing

# Exemple d'utilisation
text = "Votre texte ici..."
normalized = pre_processing.normalizing(text)
tokens = pre_processing.spacy_tokenization(normalized)
lemmas = pre_processing.spacy_lemmatization(normalized)
```

## 📂 Structure du projet

```
npl_mi3a/
├── pre_processing.py    # Fichier principal avec toutes les fonctions (À RENDRE)
├── demo.py             # Fichier de démonstration alternatif
├── main.py             # Point d'entrée original
├── README.md           # Ce fichier
├── pyproject.toml      # Configuration du projet
└── uv.lock            # Fichier de verrouillage des dépendances
```

## 📚 Fonctions disponibles

### Normalisation et Tokenisation
- `normalizing(text)` - Normalise le texte (minuscules, suppression de caractères spéciaux)
- `ntk_tokenization(text)` - Tokenisation avec NLTK
- `spacy_tokenization(text)` - Tokenisation avec spaCy

### Nettoyage
- `nltk_stopwords_removal(tokens)` - Supprime les stopwords avec NLTK
- `spacy_stopwords_removal(text)` - Supprime les stopwords avec spaCy
- `remove_punctuation(tokens)` - Supprime la ponctuation

### Transformation
- `spacy_lemmatization(text)` - Lemmatisation avec spaCy
- `stemming_vs_lemmatization(text)` - Compare stemming et lemmatisation

### Analyse
- `word_frequencies(tokens, top_n)` - Calcule les fréquences des mots
- `frequency_analysis(tokens)` - Analyse fréquences absolues vs relatives
- `average_word_length(tokens)` - Calcule la longueur moyenne des mots
- `vocabulary_size(tokens)` - Analyse la taille du vocabulaire
- `extract_ngrams(tokens, n)` - Extrait les n-grams

### Avancé
- `named_entity_recognition(text)` - Reconnaissance d'entités nommées (NER)
- `preprocessing_pipeline(text)` - Pipeline complet de traitement

## 📊 Exemple de sortie

Lorsque vous exécutez `python pre_processing.py`, vous obtiendrez:

```
================================================================================
       DÉMONSTRATION DES FONCTIONS DE PRÉTRAITEMENT NLP
================================================================================

[TEST 1] Normalisation du texte
--------------------------------------------------------------------------------
Texte original =>:
[texte affiché]

Après normalisation =>:
[texte normalisé affiché]

[... tous les tests ...]

================================================================================
                         RÉSUMÉ DES RÉSULTATS
================================================================================

  📊 Statistiques générales:
    • Texte normalisé: XXX caractères
    • Tokens NLTK: XX
    • Tokens spaCy: XX
    • Tokens après stopwords (NLTK): XX
    • Lemmes: XX

  🏷️  Entités nommées:
    • Nombre d'entités détectées: X

  📖 Analyse du vocabulaire:
    • Mots uniques: XX
    • Longueur moyenne: X.XX caractères/mot
    • Ratio vocabulaire/total: XX.XX%

  🔤 N-grams:
    • Bi-grams générés: XX
    • Tri-grams générés: XX

================================================================================
       ✅ TOUTES LES FONCTIONS ONT ÉTÉ TESTÉES AVEC SUCCÈS
================================================================================
```

## 📖 Bibliothèques utilisées

- **NLTK** (Natural Language Toolkit) - Tokenisation, stopwords, stemming, n-grams
- **spaCy** (fr_core_news_sm) - Tokenisation, lemmatisation, NER, stopwords
- **Python Standard Library** - re, string, collections

## 👨‍💻 Auteur

MI3A - TP NLP 2025

## 📄 Licence

Projet académique - TP1 Intelligence Artificielle
