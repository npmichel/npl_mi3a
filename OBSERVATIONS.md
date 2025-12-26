# Observations et Réponses aux Questions du TP

## Question 2 bis - Comparaison NLTK vs spaCy pour la tokenisation

**Remarques ntk et spacy**

En exécutant les deux fonctions `ntk_tokenization()` et `spacy_tokenization()` sur notre corpus:

- **Nombre de tokens identique**: Les deux bibliothèques produisent 117 tokens
- **Résultats très similaires**: Pour le texte normalisé, NLTK et spaCy segmentent de manière identique
- **Différence principale**:
  - NLTK utilise un algorithme basé sur des règles linguistiques
  - spaCy utilise un modèle de deep learning entraîné sur des corpus français

**Conclusion**: Pour un texte déjà normalisé en français, les deux approches donnent des résultats équivalents. La différence se verrait plus sur du texte non normalisé ou avec des cas complexes.

---

## Question 3 bis - Comparaison NLTK vs spaCy pour les stopwords

**Observations**:

- **NLTK**: 75 tokens restants après suppression (42 stopwords supprimés)
- **spaCy**: 70 tokens restants après suppression (47 stopwords supprimés)
- **Différence**: spaCy est plus strict et supprime 5 tokens de plus que NLTK

**Eléments de réponse?**

- Les deux bibliothèques ont des listes de stopwords différentes
- spaCy inclut des mots comme "très", "certains", "cependant" dans sa liste
- NLTK conserve certains mots que spaCy considère comme stopwords

---

## Question 6 - Reconnaissance d'entités nommées (NER)

**Entités détectées par spaCy**:

1. "Coupe d'Afrique des Nations 2025" → MISC (événement)
2. "CAN" → ORG (organisation)
3. "Maroc" → LOC (lieu) - 2 fois
4. "Casablanca", "Rabat", "Marrakech", "Fès", "Tanger" → LOC (villes)
5. "Lions de l'Atlas" → ORG (équipe)
6. "Contact" → MISC (erreur de détection)

**Erreurs observées**:

- "Contact" est détecté comme MISC alors que c'est un mot commun
- "CAN" devrait être MISC (événement) plutôt que ORG
- Certaines dates (2025, décembre 2025) ne sont pas détectées comme temporelles

**Explication des erreurs**:
Le modèle `fr_core_news_sm` est un modèle léger entraîné sur des textes d'actualité généraux. Pour améliorer:

- Utiliser `fr_core_news_md` ou `fr_core_news_lg` (modèles plus précis)
- Fine-tuner le modèle sur des textes sportifs
- Post-traiter les résultats avec des règles personnalisées

---

## Question 10 - Stemming vs Lemmatisation

**Différences observées**:

### Stemming (SnowballStemmer - NLTK)

- **Approche**: Coupe les suffixes de manière algorithmique
- **Avantages**: Plus rapide, ne nécessite pas de dictionnaire
- **Inconvénients**: Peut créer des racines qui ne sont pas des mots réels
- **Exemple**: "organisateurs" → "organisat" (pas un mot français)

### Lemmatisation (spaCy)

- **Approche**: Utilise un dictionnaire et l'analyse morphologique
- **Avantages**: Produit toujours des mots valides (lemmes)
- **Inconvénients**: Plus lent, nécessite un modèle linguistique
- **Exemple**: "organisateurs" → "organisateur" (mot valide)

**Recommandation**:

- Utiliser la **lemmatisation** pour des applications nécessitant de la précision (recherche sémantique, analyse de sentiments)
- Utiliser le **stemming** pour des applications nécessitant de la rapidité (moteurs de recherche basiques, filtrage rapide)

---

## Question 14 - Fréquences absolues vs relatives

**Fréquences absolues**: Nombre brut d'occurrences d'un mot dans le texte

- Exemple: "maroc" apparaît 2 fois

**Fréquences relatives**: Proportion du mot par rapport au total des tokens

- Exemple: "maroc" = 2/70 = 0.0286 (2.86%)

**Utilité**:

- **Absolues**: Pour compter les occurrences brutes
- **Relatives**: Pour comparer des textes de tailles différentes
- Les fréquences relatives permettent de normaliser et de comparer équitablement des corpus de tailles différentes

---

## Résumé des apprentissages

1. **Normalisation**: Essentielle pour uniformiser le texte
2. **Tokenisation**: NLTK et spaCy donnent des résultats similaires pour le français
3. **Stopwords**: spaCy est plus strict que NLTK
4. **Lemmatisation > Stemming**: Pour la précision en français
5. **NER**: Nécessite des modèles spécialisés pour de meilleurs résultats
6. **N-grams**: Utiles pour capturer des expressions multi-mots
7. **Pipeline**: Automatiser le prétraitement améliore la reproductibilité
