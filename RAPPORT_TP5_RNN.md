# Rapport TP5 - Comparaison des approches Feed-Forward et RNN

## 1. Introduction

Ce rapport présente l'implémentation et la comparaison de trois approches pour la prédiction de mots : deux réseaux feed-forward avec des tailles de contexte différentes (2 et 6 mots) et un réseau de neurones récurrent (RNN).

## 2. Corpus d'étude

Le corpus utilisé contient 6 phrases avec des structures syntaxiques complexes :

```
"le chat que le chien a vu mange"
"le chien que le chat poursuit aboie"
"le chat que le voisin nourrit dort"
"le chien que le voisin adopte mange"
"le chat que le chien effraie se cache"
"le chien que le chat observe se nourrit"
```

Taille du vocabulaire : 18 mots

## 3. Architectures implémentées

### 3.1 Feed-Forward (FF-2 et FF-6)

Architecture :
- Couche Embedding (vocab_size → 8 dimensions)
- Couche Flatten
- Couche Dense (16 neurones, activation ReLU)
- Couche de sortie (vocab_size neurones, activation softmax)

Paramètres :
- Taille de contexte : 2 mots (FF-2) ou 6 mots (FF-6)
- Époques : 200
- Optimiseur : Adam
- Fonction de perte : sparse_categorical_crossentropy

### 3.2 RNN

Architecture :
- Couche Embedding (vocab_size → 8 dimensions)
- Couche SimpleRNN (16 unités)
- Couche Dense (16 neurones, activation ReLU)
- Couche de sortie (vocab_size neurones, activation softmax)

Paramètres :
- Séquences de longueur variable (avec padding)
- Époques : 200
- Optimiseur : Adam
- Fonction de perte : sparse_categorical_crossentropy

## 4. Résultats expérimentaux

### 4.1 Feed-Forward avec contexte de 2 mots (FF-2)

Contexte testé : ['le', 'chat']

Résultats :
- Mot prédit : que
- Top 3 : ['que', 'chien', 'a']
- Probabilités : [0.982, 0.004, 0.002]

### 4.2 Feed-Forward avec contexte de 6 mots (FF-6)

Contexte testé : ['le', 'chat', 'que', 'le', 'chien', 'a']

Résultats :
- Mot prédit : vu
- Top 3 : ['vu', 'dort', 'aboie']
- Probabilités : [0.478, 0.130, 0.114]

### 4.3 RNN

Contexte testé : ['le', 'chat', 'que', 'le', 'chien', 'a', 'vu']

Résultats :
- Mot prédit : mange
- Top 3 : ['mange', 'cache', 'effraie']
- Probabilités : [0.848, 0.078, 0.017]

## 5. Analyse comparative

### 5.1 Tableau récapitulatif

| Modèle | Taille contexte | Contexte | Prédiction | Confiance |
|--------|----------------|----------|------------|-----------|
| FF-2 | 2 mots | le chat | que | 98.2% |
| FF-6 | 6 mots | le chat que le chien a | vu | 47.8% |
| RNN | Variable | le chat que le chien a vu | mange | 84.8% |

### 5.2 Interprétation des résultats

FF-2 : Avec seulement 2 mots de contexte, le modèle prédit "que" car la séquence "le chat que" apparaît fréquemment dans le corpus (4 fois sur 6 phrases). La prédiction est très confiante (98.2%).

FF-6 : Avec un contexte plus large, le modèle prédit "vu" pour compléter la construction "a vu" observée dans la première phrase du corpus. La confiance est modérée (47.8%) car le modèle feed-forward n'a pas de mémoire séquentielle et traite le contexte comme un vecteur fixe.

RNN : Le RNN prédit "mange" avec une confiance élevée (84.8%). Contrairement aux feed-forward, le RNN maintient un état caché qui lui permet de garder en mémoire que "le chat" est le sujet principal de la phrase, malgré la proposition relative intermédiaire. Il associe correctement le sujet "chat" au verbe "mange" qui apparaît dans les phrases du corpus avec ce sujet.

### 5.3 Différences fondamentales

Gestion du contexte :
- FF-2 et FF-6 utilisent une fenêtre fixe, ils ne capturent que des patterns locaux
- RNN utilise son état caché pour mémoriser l'information sur toute la séquence

Dépendances longues :
- FF-2 ne voit que les 2 derniers mots
- FF-6 voit 6 mots mais sans notion d'ordre temporel
- RNN peut théoriquement capturer des dépendances à n'importe quelle distance grâce à sa récurrence

Performance observée :
- FF-2 : Excellente confiance mais contexte très limité
- FF-6 : Contexte plus large mais prédiction moins confiante
- RNN : Bonne confiance et meilleure compréhension de la structure globale de la phrase

## 6. Question 12 : Comparaison RNN vs Feed-Forward

Le RNN et le feed-forward diffèrent principalement dans leur façon de traiter l'information séquentielle.

Le feed-forward traite son contexte comme un vecteur statique. FF-6 voit "le chat que le chien a" mais ne comprend pas la structure hiérarchique de la phrase. Il se concentre sur le pattern local "a" et prédit logiquement "vu" pour compléter "a vu".

Le RNN, grâce à sa couche récurrente, traite la séquence mot par mot en mettant à jour son état caché à chaque étape. Quand il lit "le chat que le chien a vu", il maintient l'information que "le chat" est le sujet principal et prédit donc "mange", une action cohérente avec ce sujet observée dans le corpus.

Cette différence explique pourquoi :
- FF-6 prédit "vu" (continuation de la proposition relative)
- RNN prédit "mange" (verbe principal lié au sujet "chat")

Les deux prédictions sont linguistiquement valides mais reflètent une compréhension différente de la structure de la phrase.

## 7. Conclusion

Cette comparaison montre que le choix entre feed-forward et RNN dépend de la tâche :
- Pour des prédictions locales simples : feed-forward suffit
- Pour capturer des dépendances longues et des structures complexes : RNN est préférable

Le RNN démontre une meilleure capacité à comprendre la structure syntaxique globale, même si cela se fait au prix d'une complexité computationnelle plus élevée.
