# Rapport TP : Réseau Neuronal Feed-Forward pour la Prédiction de Mots

## Question 7 : Généralisation du modèle feed-forward

### Contexte
Après avoir entraîné le réseau feed-forward sur notre corpus, nous testons sa capacité à généraliser en prédisant le mot suivant le contexte ["le", "chien"]. Le modèle prédit les trois mots les plus probables : "se", "mange", et "aboie".

### Observations

Les résultats de l'exécution montrent :
- Mot le plus probable : "se"
- Top 3 des mots prédits : ['se', 'mange', 'aboie']

### Analyse

Le modèle démontre une capacité de généralisation intéressante. Bien que le réseau ait été entraîné sur seulement 6 phrases, il parvient à extraire les trois mots qui apparaissent effectivement après "le chien" dans le corpus d'entraînement :
- "le chien mange" (présent dans les données)
- "le chien se nourrit" (présent dans les données)
- "le chien aboie" (présent dans les données)

Cette prédiction n'est pas aléatoire : le modèle a appris à associer le contexte ["le", "chien"] avec les mots qui suivent habituellement. Le fait que "se" soit prédit en premier peut s'expliquer par la fréquence d'apparition ou par les patterns appris lors de l'entraînement.

La couche d'embeddings (de dimension 8) permet au modèle de capturer des représentations vectorielles des mots, tandis que la couche Dense cachée avec activation ReLU permet d'apprendre des relations non-linéaires entre les contextes et les mots cibles.

## Question 8 : Prédiction avec les n-grammes

### Comparaison avec le modèle bigramme

Le code compare le réseau feed-forward avec un modèle bigramme traditionnel. Pour le contexte "chien", le modèle bigramme calcule P(mange | chien) = 0.333.

### Analyse comparative

**Modèle n-gramme (bigramme) :**
- Calcule des probabilités basées uniquement sur des fréquences de co-occurrence
- P(mange | chien) = 0.333 signifie qu'il y a 33.3% de chance que "mange" suive "chien"
- Approche purement statistique, déterministe
- Ne considère que les paires de mots vues pendant l'entraînement

**Modèle feed-forward :**
- Prédit ['se', 'mange', 'aboie'] pour le contexte ["le", "chien"]
- Utilise des embeddings pour représenter les mots dans un espace vectoriel continu
- Peut capturer des relations sémantiques au-delà de simples co-occurrences
- La couche cachée permet d'apprendre des patterns plus complexes

### Observations

Le bigramme donne une probabilité de 33.3% pour "mange" après "chien", ce qui est cohérent avec le corpus (1 occurrence de "chien mange" sur 3 occurrences totales de "chien"). Cependant, le modèle feed-forward classe "se" en premier, suivi de "mange" et "aboie". Cette différence s'explique par le fait que le réseau neuronal considère un contexte de 2 mots (["le", "chien"]) alors que le bigramme ne considère qu'un seul mot ("chien").

## Question 9 : Comparaison finale

### Pourquoi le réseau feed-forward produit une prédiction plus pertinente que le modèle n-grammes ?

**1. La notion d'embeddings**

Les embeddings transforment les mots en vecteurs denses de dimension fixe (8 dans notre cas). Cette représentation permet :
- De capturer des similarités sémantiques entre les mots dans un espace vectoriel continu
- D'apprendre automatiquement des représentations qui encodent des informations linguistiques
- De réduire la dimensionnalité par rapport à une représentation one-hot
- De permettre des opérations mathématiques significatives sur les mots (addition, soustraction de vecteurs)

Contrairement aux n-grammes qui traitent chaque mot comme une unité discrète et indépendante, les embeddings permettent au modèle de comprendre que "chat" et "chien" sont sémantiquement similaires (tous deux sont des animaux) et peuvent donc partager des comportements similaires.

**2. La similarité sémantique**

Grâce aux embeddings, le réseau peut apprendre que :
- "chat" et "chien" sont similaires (animaux domestiques)
- Ils partagent des actions communes ("mange", "se nourrit")
- Les patterns observés pour "chat" peuvent s'appliquer à "chien" et vice-versa

Par exemple, si le modèle apprend que "le chat mange" et "le chat se nourrit" sont des constructions valides, il peut généraliser cette connaissance à "le chien mange" et "le chien se nourrit", même si ces combinaisons sont moins fréquentes dans le corpus.

Les n-grammes, en revanche, ne peuvent pas effectuer cette généralisation sémantique : ils ne "savent" pas que "chat" et "chien" sont similaires et traitent chaque bigramme indépendamment.

**3. La capacité de généralisation**

Le réseau feed-forward généralise mieux pour plusieurs raisons :

**Contexte étendu :** Notre modèle utilise 2 mots de contexte, permettant de capturer des dépendances plus longues que les bigrammes simples.

**Apprentissage de patterns non-linéaires :** La couche Dense avec activation ReLU permet au modèle d'apprendre des relations complexes et non-linéaires entre les contextes et les mots cibles. Les n-grammes sont limités à des statistiques de co-occurrence linéaires.

**Partage de représentations :** Les embeddings partagés permettent au modèle de transférer la connaissance d'un contexte à un autre. Si le modèle apprend que ["le", "chat"] est souvent suivi de "mange", il peut appliquer cette connaissance à ["le", "chien"] grâce à la similarité des embeddings.

**Robustesse à la rareté des données :** Les n-grammes souffrent du problème de sparsité : si une séquence n'a jamais été vue, sa probabilité est nulle. Le réseau neuronal, grâce aux embeddings et à sa capacité de généralisation, peut prédire des séquences plausibles même si elles n'ont pas été observées exactement sous cette forme.

**Lissage implicite :** Contrairement aux n-grammes qui nécessitent des techniques de lissage explicites (Laplace, Kneser-Ney), le réseau neuronal effectue un lissage implicite grâce à sa structure et à la régularisation pendant l'entraînement.

### Limitations et considérations

Bien que le réseau feed-forward soit plus performant que les n-grammes simples, il présente certaines limitations :
- Nécessite plus de données pour bien entraîner les embeddings
- Plus coûteux en calcul que les simples comptages de n-grammes
- Peut sur-apprendre sur de petits corpus (comme le nôtre avec seulement 6 phrases)
- La taille du contexte est fixe (2 mots), contrairement aux modèles plus avancés comme les LSTM ou Transformers

Dans notre expérimentation avec un corpus très limité, le modèle parvient tout de même à capturer les patterns essentiels et à prédire les trois mots qui apparaissent effectivement après "le chien" dans le corpus d'entraînement, démontrant sa capacité de généralisation même dans des conditions d'apprentissage contraintes.
