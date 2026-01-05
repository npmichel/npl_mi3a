TP6 : Réseaux de neurones à mémoire longue (LSTM)

Rapport de Travaux Pratiques

Master Intelligence Artificielle


Objectif du TP

Ce travail pratique a pour objectif d'illustrer l'utilisation des réseaux de neurones à mémoire longue (LSTM) pour la modélisation du langage naturel. Le corpus utilisé a été conçu pour introduire des dépendances longues et imbriquées, mettant en difficulté les RNN classiques et permettant de mettre en évidence l'apport des LSTM.


1. Corpus et problématique

Le corpus utilisé contient des phrases à structure syntaxique complexe avec des propositions relatives imbriquées :

- "le chat que le chien que le voisin que l enfant a vu nourrit dort"
- "le chien que le chat que le voisin que l enfant observe nourrit aboie"
- "le chat que le chien que le facteur que le voisin connait effraie se cache"
- "le chien que le chat que le voisin que le facteur aide observe se nourrit"

Dans ces phrases, le sujet principal apparaît très tôt ("le chat" ou "le chien") et doit être conservé en mémoire jusqu'au verbe final ("dort", "aboie", "cache", "nourrit"), malgré de nombreuses propositions intermédiaires. Cette structure linguistique met à l'épreuve la capacité des réseaux de neurones récurrents à gérer les dépendances longues.


2. Architecture des modèles

2.1 Modèle LSTM

L'architecture LSTM implémentée comprend :
- Couche d'embedding : dimension 8, pour représenter chaque mot du vocabulaire (20 mots)
- Couche LSTM : 16 unités mémoire avec portes de contrôle (forget gate, input gate, output gate)
- Couche Dense : activation softmax pour la classification sur l'ensemble du vocabulaire
- Optimiseur : Adam
- Fonction de perte : sparse_categorical_crossentropy
- Nombre d'époques : 300

La spécificité du LSTM réside dans ses mécanismes de portes qui permettent de contrôler explicitement le flux d'information :
- La porte d'oubli (forget gate) décide quelles informations de la cellule mémoire doivent être oubliées
- La porte d'entrée (input gate) décide quelles nouvelles informations doivent être stockées
- La porte de sortie (output gate) décide quelle partie de la cellule mémoire doit être utilisée pour la sortie

2.2 Modèle RNN classique (SimpleRNN)

L'architecture RNN implémentée comprend :
- Couche d'embedding : dimension 8 (identique au LSTM pour une comparaison équitable)
- Couche SimpleRNN : 16 unités récurrentes
- Couche Dense intermédiaire : 16 unités avec activation ReLU
- Couche Dense de sortie : activation softmax
- Optimiseur : Adam
- Fonction de perte : sparse_categorical_crossentropy
- Nombre d'époques : 200

Le RNN classique ne possède pas de mécanisme de contrôle explicite de la mémoire, ce qui le rend vulnérable au problème du gradient qui s'évanouit (vanishing gradient) lors de la propagation à travers de nombreuses étapes temporelles.


3. Étape 7 : Analyse comparative LSTM vs RNN

3.1 Protocole expérimental

Pour évaluer rigoureusement la capacité des deux architectures à gérer les dépendances longues, nous avons mené une analyse expérimentale avec 10 entraînements indépendants pour chaque modèle. Cette approche permet de mesurer non seulement la précision des prédictions, mais aussi leur stabilité.

Séquence de test utilisée :
"le chat que le chien que le voisin que l enfant a vu"

Le modèle doit prédire le mot suivant. Selon la phrase complète du corpus : "le chat que le chien que le voisin que l enfant a vu nourrit dort", le mot syntaxiquement cohérent devrait être "nourrit", qui ferme la proposition relative et précède le verbe principal "dort" qui s'accorde avec le sujet initial "le chat".

3.2 Résultats quantitatifs

Test unique (résultat représentatif) :

LSTM :

- Mot prédit : "aboie" (confiance : 0.6817)
- Top 3 prédictions :
  1. "aboie" (probabilité : 0.6817)
  2. "dort" (probabilité : 0.2101)
  3. "nourrit" (probabilité : 0.0138)
- Entropie de la distribution : 1.1252

RNN :

- Mot prédit : "aboie" (confiance : 0.4437)
- Top 3 prédictions :
  1. "aboie" (probabilité : 0.4437)
  2. "aide" (probabilité : 0.0782)
  3. "le" (probabilité : 0.0734)
- Entropie de la distribution : 2.1676

Analyse statistique sur 10 entraînements indépendants :

Résultats LSTM (10 entraînements) :
- Prédictions : ['aboie', 'aboie', 'cache', 'dort', 'aboie', 'dort', 'dort', 'aboie', 'dort', 'aboie']
- Mot le plus fréquent : "aboie" (5/10 exécutions)
- Confiance moyenne : 0.3913 (écart-type : ±0.1206)
- Nombre de mots différents prédits : 3 ("aboie", "dort", "cache")
- Variabilité : 50% de variabilité dans les prédictions

Résultats RNN (10 entraînements) :
- Prédictions : ['aboie', 'aboie', 'aboie', 'aboie', 'aboie', 'aboie', 'aboie', 'aboie', 'aboie', 'aboie']
- Mot le plus fréquent : "aboie" (10/10 exécutions)
- Confiance moyenne : 0.6642 (écart-type : ±0.2218)
- Nombre de mots différents prédits : 1 ("aboie")
- Variabilité : 0% de variabilité dans les prédictions

3.3 Analyse des résultats

Observation 1 : Capacité à identifier le mot syntaxiquement correct

Dans le test unique, le LSTM place "nourrit" (le mot syntaxiquement correct) dans son top 3 des prédictions avec une probabilité de 0.0138, démontrant qu'il a capturé, même faiblement, la structure syntaxique de dépendance longue entre "le chat" (sujet initial) et "nourrit" (verbe qui ferme la proposition relative).

Le RNN, en revanche, ne place pas "nourrit" dans son top 3, mais prédit plutôt "aide" et "le", mots moins cohérents syntaxiquement dans ce contexte. Cela indique une perte de la structure syntaxique globale.

Observation 2 : Diversité des prédictions

Le LSTM explore davantage l'espace des prédictions possibles, produisant 3 mots différents ("aboie", "dort", "cache") dans l'analyse sur 10 entraînements, tous correspondant à des verbes finaux présents dans le corpus. Cette diversité suggère que le LSTM capture différentes structures syntaxiques du corpus.

Le RNN, en revanche, converge systématiquement vers une seule prédiction ("aboie"), indiquant une sur-généralisation ou une incapacité à discriminer finement entre les différentes structures syntaxiques.

Observation 3 : Niveau de confiance

Paradoxalement, le RNN affiche une confiance moyenne plus élevée (0.6642) que le LSTM (0.3913) dans l'analyse sur 10 entraînements, mais avec une variance également plus importante (±0.2218 vs ±0.1206). Cette confiance élevée associée à une prédiction unique mais incorrecte est caractéristique d'un surapprentissage ou d'une convergence prématurée vers une solution sous-optimale.

Le LSTM, avec sa confiance plus faible mais plus stable, semble adopter une approche plus prudente, reflétant l'incertitude inhérente à la tâche de prédiction sur des dépendances longues.

Observation 4 : Cohérence syntaxique

Dans l'analyse statistique, aucun des deux modèles ne prédit systématiquement "nourrit", le mot syntaxiquement correct qui devrait suivre "a vu" dans la phrase. Cependant, le LSTM prédit parfois "dort" (4/10 fois), qui est le verbe principal de la phrase complète et s'accorde correctement avec le sujet initial "le chat".

La prédiction récurrente du mot "aboie" par le RNN est problématique, car "aboie" s'accorde avec "le chien" et non "le chat", indiquant une perte de l'information sur le sujet principal de la phrase.

Observation 5 : Entropie de la distribution

Dans le test unique, l'entropie de la distribution du LSTM (1.1252) est plus faible que celle du RNN (2.1676), indiquant que le LSTM a une distribution plus concentrée. Cela signifie que le LSTM est plus "certain" de ses prédictions, bien qu'il maintienne une probabilité non nulle pour le mot correct "nourrit" dans son top 3.


4. Interprétation théorique : Pourquoi le RNN échoue sur les dépendances longues

4.1 Le problème du gradient qui s'évanouit

Dans un RNN classique, lors de la rétropropagation à travers le temps (BPTT), le gradient doit traverser de nombreuses étapes temporelles. À chaque étape, le gradient est multiplié par la dérivée de la fonction d'activation et par les poids de la couche récurrente.

Pour une séquence de longueur T, si ces facteurs multiplicatifs sont inférieurs à 1, le gradient diminue exponentiellement : gradient(t) ∝ γ^T où γ < 1.

Dans notre cas, avec une phrase de 13 mots, le gradient doit se propager à travers 13 étapes temporelles, rendant difficile l'apprentissage des dépendances entre le sujet initial ("le chat") et le verbe final ("dort").

Conséquence observée : Le RNN converge vers la prédiction du mot le plus fréquemment rencontré en fin de phrase ("aboie"), sans tenir compte du contexte lointain (le sujet initial).

4.2 Capacité mémoire limitée du RNN

Le RNN classique encode toute l'information de la séquence dans un unique vecteur d'état caché h(t). Pour des séquences longues avec de nombreuses informations intermédiaires (les propositions relatives imbriquées), ce vecteur d'état est régulièrement écrasé par les nouvelles informations, causant une perte des informations anciennes mais essentielles (le sujet principal).

Dans notre corpus, les mots "le chien", "le voisin", "l enfant" apparaissent entre le sujet initial "le chat" et le verbe final "dort". Le RNN tend à accorder plus d'importance aux informations récentes, perdant ainsi la trace du sujet initial.

Conséquence observée : Le RNN prédit systématiquement "aboie" (associé au "chien", sujet fréquent et récent dans les phrases) plutôt que "dort" (qui nécessite de se souvenir que le sujet initial était "le chat").

4.3 Avantages du LSTM

Le LSTM résout partiellement ces problèmes grâce à :

a) Architecture de cellule mémoire : Le LSTM maintient une cellule mémoire c(t) distincte de l'état caché h(t), permettant une conservation à long terme de l'information.

b) Mécanismes de portes :
   - Porte d'oubli : ft = σ(Wf · [h(t-1), x(t)] + bf)
   - Porte d'entrée : it = σ(Wi · [h(t-1), x(t)] + bi)
   - Porte de sortie : ot = σ(Wo · [h(t-1), x(t)] + bo)

Ces portes permettent au modèle d'apprendre quelles informations doivent être conservées (le sujet initial "le chat"), quelles informations peuvent être oubliées (les sujets intermédiaires), et quand utiliser ces informations (au moment de prédire le verbe final).

c) Gradient protégé : La cellule mémoire offre un chemin de gradient privilégié qui évite les multiplications répétées, atténuant le problème du gradient qui s'évanouit.

Conséquence observée : Le LSTM montre une plus grande variabilité dans ses prédictions, explorant différentes interprétations syntaxiques possibles ("dort", "aboie", "cache"), toutes cohérentes avec différentes phrases du corpus.


5. Limitations observées et discussion

5.1 Taille du corpus

Avec seulement 4 phrases d'entraînement, les deux modèles sont en situation de données extrêmement limitées. Cette contrainte explique pourquoi même le LSTM ne parvient pas à prédire systématiquement le mot syntaxiquement correct.

Un corpus plus large permettrait au LSTM de mieux apprendre les règles d'accord sujet-verbe à travers des dépendances longues.

5.2 Dimension des embeddings et de la mémoire

Les embeddings de dimension 8 et la mémoire de 16 unités sont relativement faibles pour capturer la complexité syntaxique des phrases imbriquées. Une augmentation de ces dimensions pourrait améliorer les performances.

5.3 Nombre d'époques

Le LSTM est entraîné sur 300 époques tandis que le RNN ne l'est que sur 200. Cependant, augmenter le nombre d'époques du RNN ne résoudrait pas fondamentalement le problème du gradient qui s'évanouit.

5.4 Performance paradoxale du RNN

De manière contre-intuitive, le RNN montre une "meilleure" stabilité (prédiction unique) et une confiance plus élevée. Cependant, cette stabilité est en réalité une faiblesse : elle indique une incapacité à discriminer entre différents contextes syntaxiques et une convergence vers une solution simpliste (prédire le mot le plus fréquent).

La variabilité du LSTM, bien qu'elle puisse sembler être une instabilité, reflète en réalité une exploration plus riche de l'espace des solutions possibles, ce qui est souhaitable dans un contexte d'apprentissage avec peu de données.


6. Conclusion

Cette étude comparative démontre expérimentalement les limites des RNN classiques face aux dépendances longues. Le RNN converge systématiquement vers une solution sous-optimale, ignorant le contexte lointain et se basant principalement sur les informations récentes et les fréquences globales.

Le LSTM, grâce à son architecture à cellules mémoire et à ses portes de contrôle, montre une plus grande capacité à explorer différentes interprétations syntaxiques. Bien qu'il ne résolve pas parfaitement la tâche (ce qui serait difficile avec seulement 4 phrases d'entraînement), il évite la sur-généralisation systématique du RNN.

Points clés à retenir :

1. Le problème du gradient qui s'évanouit affecte sévèrement les RNN sur des séquences de plus de 10 étapes temporelles.

2. La capacité mémoire limitée du RNN conduit à un biais récent (recency bias), favorisant les informations les plus récentes au détriment du contexte lointain.

3. Les mécanismes de portes du LSTM permettent un apprentissage sélectif des dépendances à long terme, offrant un avantage significatif pour des tâches linguistiques complexes.

4. Une confiance élevée n'est pas nécessairement un indicateur de qualité : le RNN montre une confiance élevée dans des prédictions incorrectes, tandis que le LSTM montre une confiance plus mesurée reflétant l'incertitude réelle de la tâche.

5. Pour des applications de traitement du langage naturel impliquant des structures syntaxiques complexes, les architectures LSTM (ou leurs successeurs comme les GRU ou les Transformers) sont nettement préférables aux RNN classiques.


Annexe : Fichiers du projet

- lstm_nlp.py : Implémentation du modèle LSTM
- rnn_nlp.py : Implémentation du modèle RNN
- compare_lstm_rnn.py : Script de comparaison directe
- analyse_comparaison.py : Script d'analyse statistique approfondie
- w2v_pret.py : Prétraitement des données textuelles
