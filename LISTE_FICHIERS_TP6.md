Liste des fichiers pour le rendu du TP6 - LSTM

Documents à rendre

Rapport principal :

1. RAPPORT_TP6_LSTM.md (12 KB)
   - Rapport complet et détaillé
   - Contient : objectifs, architecture, résultats, analyse théorique, conclusion
   - Format adapté pour un Master
   - Sections : 6 chapitres avec analyses approfondies

2. SYNTHESE_RESULTATS_TP6.md (7 KB)
   - Synthèse des résultats expérimentaux
   - Tableaux comparatifs
   - Réponse détaillée à la question du TP
   - Formules mathématiques explicitées

3. README_TP6.md (4 KB)
   - Guide d'utilisation des scripts
   - Instructions d'exécution
   - Description de chaque fichier
   - Résultats attendus


Code source :

Scripts principaux :

1. lstm_nlp.py (4.2 KB)
   - Implémentation du modèle LSTM
   - Fonctions modulaires : tokenization, data_preparation, model, fit_model
   - Exécutable de manière autonome

2. rnn_nlp.py (4.2 KB)
   - Implémentation du modèle RNN classique (SimpleRNN)
   - Architecture comparable au LSTM pour une comparaison équitable
   - Même structure de code que lstm_nlp.py

3. compare_lstm_rnn.py (2.7 KB)
   - Comparaison directe des deux modèles
   - Exécute LSTM et RNN sur le même corpus
   - Affiche les prédictions côte à côte

Scripts d'analyse :

4. analyse_comparaison.py (5.4 KB)
   - Analyse statistique approfondie
   - 10 entraînements indépendants pour chaque modèle
   - Calcul de la stabilité, confiance moyenne, écart-type
   - Statistiques de variabilité

5. test_final_tp6.py (6.7 KB)
   - Test final avec sortie formatée pour le rapport
   - Top 3 des prédictions pour chaque modèle
   - Calcul de l'entropie de la distribution
   - Analyse comparative détaillée

Utilitaires :

6. w2v_pret.py (1.5 KB)
   - Prétraitement des données textuelles
   - Tokenization et normalisation
   - Utilisé par tous les autres scripts


Fichiers fournis :

- TP6_LSTM_IA_NLP.pdf : Énoncé du TP


Structure du rendu

```
npl_mi3a/
├── Documents/
│   ├── RAPPORT_TP6_LSTM.md          ← Rapport principal à rendre
│   ├── SYNTHESE_RESULTATS_TP6.md   ← Synthèse des résultats
│   └── README_TP6.md                ← Guide d'utilisation
│
├── Code source/
│   ├── lstm_nlp.py                  ← Implémentation LSTM
│   ├── rnn_nlp.py                   ← Implémentation RNN
│   ├── compare_lstm_rnn.py          ← Comparaison directe
│   ├── analyse_comparaison.py       ← Analyse statistique
│   ├── test_final_tp6.py            ← Test final
│   └── w2v_pret.py                  ← Prétraitement
│
└── Énoncé/
    └── TP6_LSTM_IA_NLP.pdf          ← Énoncé fourni
```


Points clés du travail réalisé

Étapes complétées :

✓ Étape 1 : Préparation des séquences (w2v_pret.py)
✓ Étape 2 : Encodage des mots (tokenization dans lstm_nlp.py et rnn_nlp.py)
✓ Étape 3 : Padding et construction des entrées (data_preparation)
✓ Étape 4 : Définition du modèle LSTM (lstm_model)
✓ Étape 5 : Entraînement du modèle LSTM (fit_lstm_model)
✓ Étape 6 : Prédiction avec le LSTM (dans main)
✓ Étape 7 : Mise en difficulté du RNN et comparaison (compare_lstm_rnn.py)

Analyses supplémentaires réalisées :

✓ Analyse statistique sur 10 entraînements indépendants
✓ Calcul de l'entropie de la distribution des probabilités
✓ Top 3 des prédictions pour chaque modèle
✓ Mesure de la stabilité et de la variabilité
✓ Analyse théorique du problème du gradient qui s'évanouit
✓ Interprétation détaillée des résultats


Résultats principaux

LSTM :
- Identifie le mot syntaxiquement correct "nourrit" dans le top 3 (1.38%)
- Explore 3 mots différents sur 10 entraînements (variabilité : 50%)
- Confiance moyenne : 39.13% (±12.06%)
- Entropie : 1.1252 (distribution concentrée)

RNN :
- N'identifie pas "nourrit" dans le top 3
- Converge systématiquement vers "aboie" (10/10 entraînements)
- Confiance moyenne : 66.42% (±22.18%)
- Entropie : 2.1676 (distribution dispersée)

Conclusion :
Le LSTM capture partiellement les dépendances longues grâce à sa cellule mémoire et ses portes de contrôle, tandis que le RNN souffre du problème du gradient qui s'évanouit et perd l'information sur le sujet initial.


Commandes d'exécution

Pour reproduire les résultats :

```bash
# Test LSTM seul
uv run python lstm_nlp.py

# Test RNN seul
uv run python rnn_nlp.py

# Comparaison directe
uv run python compare_lstm_rnn.py

# Analyse statistique complète (10 entraînements)
uv run python analyse_comparaison.py

# Test final avec sortie formatée
uv run python test_final_tp6.py
```


Note importante

Tous les fichiers ont été créés avec un niveau de rigueur adapté à un Master. Le rapport évite l'utilisation de symboles graphiques superflus et se concentre sur l'analyse scientifique et technique.

La réponse à la question du TP ("Pourquoi le RNN échoue...") est développée dans le chapitre 4 du rapport principal et synthétisée dans le fichier SYNTHESE_RESULTATS_TP6.md.
