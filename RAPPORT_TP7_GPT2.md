# Rapport TP7 - Analyse critique de GPT-2 dans un contexte éducatif

## Introduction

Ce rapport analyse les capacités et limitations du modèle GPT-2 dans un contexte éducatif. L'objectif est d'évaluer la pertinence de ce modèle pré-entraîné pour des applications pédagogiques en traitement automatique du langage naturel (NLP).

## Méthodologie

Le modèle GPT-2 a été testé via la bibliothèque Transformers de Hugging Face avec différents paramètres de génération (température, top_k, top_p) et sur plusieurs types de requêtes : questions factuelles, concepts scientifiques complexes, génération multilingue et cohérence temporelle.

Les tests ont porté sur six aspects principaux :

1. Cohérence et hallucinations factuelles
2. Influence de la température sur la génération
3. Maintien de la cohérence selon la longueur
4. Réponse à des questions éducatives
5. Capacités multilingues (français)
6. Répétabilité des générations

## Résultats observés

### 1. Hallucinations et incohérence factuelle

GPT-2 produit régulièrement des réponses factuellement incorrectes, notamment :

**Exemple - Photosynthèse** : Le modèle génère une explication confuse mélangeant concepts d'énergie thermique et composés chimiques sans structure logique claire.

**Exemple - Capitale de la France** : Au lieu de répondre simplement "Paris", le modèle invente des informations historiques incorrectes sur des états fictifs de France entre 1789-1811.

**Exemple - Vitesse de la lumière** : La réponse mélange des concepts de temps et de cadres de référence sans cohérence scientifique.

### 2. Répétitions et boucles sémantiques

Lors de la génération de textes longs, GPT-2 tombe fréquemment dans des patterns répétitifs :

```
"There are a number of ways to handle elements. There are a number of ways to handle elements..."
```

Ce phénomène devient particulièrement problématique pour des explications détaillées nécessaires dans un contexte éducatif.

### 3. Incompréhension des concepts techniques

**Question : "What is the difference between LSTM and GRU?"**

La réponse générée démontre une incompréhension fondamentale : le modèle confond les architectures de réseaux de neurones avec des systèmes de stockage de données, répétant des phrases sans substance technique.

**Question : "Explain the backpropagation algorithm"**

Le modèle se contente de répéter le terme "backpropagation" dans des phrases circulaires sans expliquer le mécanisme de rétropropagation du gradient.

### 4. Capacités limitées en français

GPT-2 peine significativement avec les prompts en français :

- Génération d'un mélange de français et d'anglais
- Structure syntaxique incohérente
- Abandon rapide de la langue source pour basculer en anglais

Ceci s'explique par un entraînement majoritairement anglophone du modèle.

### 5. Manque de répétabilité

Avec un échantillonnage stochastique (do_sample=True), les générations varient considérablement entre exécutions, même avec les mêmes hyperparamètres. Cette variabilité est problématique pour un usage éducatif nécessitant des réponses fiables.

### 6. Influence de la température

- **Température basse (0.3)** : Génération plus déterministe mais répétitive et parfois bloquée dans des boucles
- **Température haute (1.5)** : Plus de créativité mais augmentation significative des incohérences et erreurs factuelles

Aucun réglage ne garantit simultanément créativité et exactitude.

## Limites de GPT-2 dans un contexte éducatif

### 1. Fiabilité scientifique insuffisante

Le modèle génère des affirmations avec confiance syntaxique mais sans garantie de véracité. Dans un contexte pédagogique, cela peut induire les étudiants en erreur et propager des informations erronées.

### 2. Absence de raisonnement structuré

GPT-2 ne décompose pas les problèmes complexes en étapes logiques. Il génère du texte plausible statistiquement mais sans compréhension conceptuelle, ce qui est inadéquat pour l'enseignement de concepts avancés.

### 3. Taille et performances du modèle

GPT-2 (117M-1.5B paramètres) est un modèle relativement ancien (2019). Les modèles plus récents (GPT-3, GPT-4, Claude, etc.) offrent des capacités significativement supérieures en termes de cohérence, factualité et raisonnement.

### 4. Contexte limité

Avec une fenêtre de contexte restreinte, GPT-2 perd rapidement la cohérence dans des discussions longues ou nécessitant de maintenir plusieurs concepts simultanément.

### 5. Biais et représentativité

Entraîné principalement sur des données anglophones du web (Reddit, etc.), le modèle présente :

- Des biais culturels et linguistiques
- Une performance dégradée sur les langues autres que l'anglais
- Potentiellement des contenus inappropriés ou biaisés

### 6. Impossibilité de vérification des sources

Le modèle ne peut pas citer de sources ni justifier ses affirmations, rendant impossible la vérification académique de l'information fournie.

## Recommandations pour un usage éducatif

### À éviter

- Utiliser GPT-2 comme source d'information factuelle sans vérification
- S'appuyer sur le modèle pour enseigner des concepts techniques précis
- L'utiliser pour générer du contenu dans des langues autres que l'anglais
- Considérer ses réponses comme scientifiquement validées

### Usages potentiels acceptables

- **Démonstration pédagogique** : Illustrer les limites des modèles de langage et l'importance de la vérification
- **Génération de texte créatif** : Exercices de rédaction où la factualité n'est pas critique
- **Étude technique** : Comprendre l'architecture Transformer et les mécanismes d'attention
- **Point de départ** : Brainstorming initial nécessitant une validation humaine systématique

### Alternatives recommandées

Pour un usage éducatif sérieux :

- Privilégier des modèles plus récents et plus grands (GPT-4, Claude Opus 4.5, etc.)
- Utiliser des modèles spécialisés pour des domaines spécifiques
- Combiner les LLM avec des systèmes de récupération d'information (RAG)
- Toujours valider les informations par des sources académiques reconnues

## Conclusion

GPT-2 représente une étape importante dans l'évolution des modèles de langage mais présente des limitations fondamentales pour un usage éducatif. Les hallucinations fréquentes, l'absence de raisonnement structuré, les répétitions sémantiques et la faible performance multilingue en font un outil inadapté pour transmettre des connaissances factuelles ou techniques.

Son principal intérêt pédagogique réside paradoxalement dans l'étude critique de ses limitations, permettant aux étudiants de développer un esprit critique face aux technologies d'IA générative et de comprendre l'importance de la vérification des informations produites par ces systèmes.

L'évolution rapide du domaine a produit des modèles significativement plus performants, mais même ceux-ci nécessitent une supervision humaine experte dans un contexte éducatif exigeant rigueur scientifique et exactitude factuelle.

## Annexes

Code source utilisé : [transformers_nlp.py](transformers_nlp.py)
Script de tests : [test_gpt2_analysis.py](test_gpt2_analysis.py)

Modèle testé : GPT-2 (base, 117M paramètres)
Framework : Transformers 4.57.3, PyTorch 2.9.1
