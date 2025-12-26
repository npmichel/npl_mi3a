# 📦 RENDU TP1 - Prétraitement NLP en Français

**Auteur**: MI3A
**Date**: Décembre 2025
**Sujet**: TP1 - Intelligence Artificielle - Traitement du Langage Naturel

---

## 📋 Contenu du rendu

### Fichiers principaux

1. **`pre_processing.py`**  **FICHIER PRINCIPAL À RENDRE**

   - Contient toutes les fonctions demandées dans le TP
   - Documentation complète en en-tête
   - Bloc `if __name__ == "__main__":` pour démonstration automatique
   - **Exécution**: `uv run python pre_processing.py`
2. **`README.md`**

   - Documentation complète du projet
   - Instructions d'installation et d'utilisation
   - Description de toutes les fonctions
3. **`OBSERVATIONS.md`**

   - Réponses aux questions du TP nécessitant des observations
   - Comparaisons NLTK vs spaCy
   - Analyse des résultats NER
   - Explications stemming vs lemmatisation
4. **`main.py`**

   - Point d'entrée alternatif
   - Exemple d'utilisation des fonctions
5. **`demo.py`**

   - Démonstration complète et détaillée
   - Alternative à l'exécution de `pre_processing.py`
6. **`pyproject.toml`** et **`uv.lock`**

   - Configuration des dépendances
   - Gestion du projet avec `uv`

---

## ✅ Questions du TP traitées

### Questions implémentées (toutes les 14 questions)

| Question | Fonction implémentée              | Statut |
| -------- | ----------------------------------- | ------ |
| Q1       | `normalizing(text)`               | ✅     |
| Q2       | `ntk_tokenization(text)`          | ✅     |
| Q2 bis   | `spacy_tokenization(text)`        | ✅     |
| Q3       | `nltk_stopwords_removal(tokens)`  | ✅     |
| Q3 bis   | `spacy_stopwords_removal(text)`   | ✅     |
| Q4       | `spacy_lemmatization(text)`       | ✅     |
| Q5       | `word_frequencies(tokens, top_n)` | ✅     |
| Q6       | `named_entity_recognition(text)`  | ✅     |
| Q8       | `preprocessing_pipeline(text)`    | ✅     |
| Q9       | `remove_punctuation(tokens)`      | ✅     |
| Q10      | `stemming_vs_lemmatization(text)` | ✅     |
| Q11      | `extract_ngrams(tokens, n)`       | ✅     |
| Q12      | `average_word_length(tokens)`     | ✅     |
| Q13      | `vocabulary_size(tokens)`         | ✅     |
| Q14      | `frequency_analysis(tokens)`      | ✅     |

---

## 🚀 Comment exécuter le TP

### Installation (une seule fois)

```bash
# Installer les dépendances
uv sync

# Vérifier que le modèle spaCy est installé
uv run python -c "import spacy; nlp = spacy.load('fr_core_news_sm'); print('✅ Modèle OK')"
```

### Exécution complète (recommandé)

```bash
# Exécuter toutes les démonstrations
uv run pre_processing.py
```

Cette commande affichera:

- Les résultats de chaque fonction sur le corpus CAN 2025
- Un résumé statistique complet
- La validation que toutes les fonctions fonctionnent

### Exécution rapide

```bash
# Exemple rapide de quelques fonctions
uv run main.py
```

### Tests individuels

```python
# Importer le module
import pre_processing

# Utiliser n'importe quelle fonction
text = "Votre texte ici..."
normalized = pre_processing.normalizing(text)
tokens = pre_processing.spacy_tokenization(normalized)
```

---

## 📊 Résultats attendus

Lors de l'exécution de `pre_processing.py`, vous devriez voir:

```
================================================================================
       DÉMONSTRATION DES FONCTIONS DE PRÉTRAITEMENT NLP
================================================================================

[TEST 1] Normalisation du texte
[TEST 2] Tokenisation avec NLTK
[TEST 2 bis] Tokenisation avec spaCy
[TEST 3] Suppression des stopwords avec NLTK
[TEST 3 bis] Suppression des stopwords avec spaCy
[TEST 4] Lemmatisation avec spaCy
[TEST 5] Calcul des fréquences des mots
[TEST 6] Reconnaissance d'entités nommées (NER)
[TEST 8] Pipeline de traitement complet
[TEST 9] Suppression de la ponctuation
[TEST 10] Comparaison Stemming vs Lemmatisation
[TEST 11] Extraction de bi-grams
[TEST 11 bis] Extraction de tri-grams
[TEST 12] Calcul de la longueur moyenne des mots
[TEST 13] Analyse de la taille du vocabulaire
[TEST 14] Analyse des fréquences absolues vs relatives

================================================================================
                         RÉSUMÉ DES RÉSULTATS
================================================================================

  📊 Statistiques générales:
    • Texte normalisé: 631 caractères
    • Tokens NLTK: 117
    • Tokens spaCy: 117
    • Tokens après stopwords (NLTK): 75
    • Lemmes: 70

  🏷️  Entités nommées:
    • Nombre d'entités détectées: 11

  📖 Analyse du vocabulaire:
    • Mots uniques: 65
    • Longueur moyenne: 6.76 caractères/mot
    • Ratio vocabulaire/total: 92.86%

  🔤 N-grams:
    • Bi-grams générés: 29
    • Tri-grams générés: 28

================================================================================
       ✅ TOUTES LES FONCTIONS ONT ÉTÉ TESTÉES AVEC SUCCÈS
================================================================================
```

---

## 🛠️ Technologies utilisées

- **Python 3.13**
- **NLTK** - Tokenisation, stopwords, stemming, n-grams
- **spaCy (fr_core_news_sm)** - Tokenisation, lemmatisation, NER, stopwords
- **uv** - Gestionnaire de paquets Python moderne

---

## 📝 Points importants

### Approche pédagogique

Chaque fonction:

- ✅ Est **unitaire** et **réutilisable**
- ✅ Affiche des **résultats détaillés** avec `print()`
- ✅ Retourne les **données traitées** pour enchaînement
- ✅ Comporte une **docstring** explicative
- ✅ Suit le **pattern établi** dans le code initial

### Comparaisons implémentées

Le TP demande de comparer différentes approches:

1. **NLTK vs spaCy** (tokenisation) → Résultats identiques sur texte normalisé
2. **NLTK vs spaCy** (stopwords) → spaCy plus strict (70 vs 75 tokens)
3. **Stemming vs Lemmatisation** → Lemmatisation plus précise pour le français
4. **Fréquences absolues vs relatives** → Relatives permettent de comparer des corpus différents

Toutes les observations sont documentées dans `OBSERVATIONS.md`.

---

## 📚 Documentation

- **README.md** - Guide complet d'utilisation
- **OBSERVATIONS.md** - Réponses aux questions d'observation
- **Docstrings** - Dans chaque fonction du code
- **Commentaires** - Explications dans le code source

---

## ✨ Bonus implémentés

Au-delà des questions du TP:

1. **Pipeline complet** (Q8) - Automatisation de toutes les étapes
2. **Démonstration automatique** - Bloc `if __name__ == "__main__"`
3. **Documentation extensive** - README, observations, commentaires
4. **Résumé statistique** - Synthèse visuelle des résultats
5. **Fonction de comparaison** - Stemming vs Lemmatisation détaillé

---

## 🎯 Conclusion

Ce TP implémente toutes les 14 questions demandées avec:

- ✅ Code fonctionnel et testé
- ✅ Documentation complète
- ✅ Observations et analyses
- ✅ Approche pédagogique (fonctions unitaires)
- ✅ Démonstration automatique

**Fichier principal à évaluer**: `pre_processing.py`

**Pour tester**: `uv run pre_processing.py`
