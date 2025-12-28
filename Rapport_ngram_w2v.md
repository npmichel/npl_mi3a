COMPARAISON N-GRAMMES vs WORD2VEC 
=====


 ## [PARTIE 1] Calcul des probabilités conditionnelles avec les bigrammes

--------------------------------------------------------------------------------
  P(chien | chat) = 0.3333 
 
  P(automobile | voiture) = 0.0000 
 
  P(transport | moyen) = 1.0000 
 


 ## [PARTIE 2] Calcul de la similarité cosinus avec Word2Vec

--------------------------------------------------------------------------------
  Sim(chat, chien) = 0.0424 

  Sim(voiture, automobile) = 0.1125 

  Sim(moyen, transport) = -0.1552 



 ## [PARTIE 3] Tableau comparatif des résultats

 ### Paire de mots             P(w2|w1) Bigrammes        Similarité Word2Vec      

--------------------------------------------------------------------------------
(chat, chien)             0.3333                    0.0424                    

(voiture, automobile)     0.0000                    0.1125                    

(moyen, transport)        1.0000                    -0.1552                   

--------------------------------------------------------------------------------


 ## [PARTIE 4] Interprétation des différences observées

--------------------------------------------------------------------------------

 ###  ANALYSE COMPARATIVE:

#### 1. DIFFÉRENCES FONDAMENTALES:
   • N-grammes: Mesure la probabilité conditionnelle P(w2|w1)

     → Basé sur la fréquence d'occurrence séquentielle dans le corpus

     → Capture l'ordre des mots et leur co-occurrence directe


   • Word2Vec: Mesure la similarité sémantique via cosinus

     → Basé sur les contextes partagés (fenêtre de mots voisins)

     → Capture les relations sémantiques, pas nécessairement la séquence


#### 2.   OBSERVATIONS SUR LES RÉSULTATS:

   • (chat, chien):

     - P(chien|chat) = 0.3333

     - Sim(chat, chien) = 0.0424

     → Probabilité 33.3%: 'chien' suit 'chat' dans certains contextes

     → Similarité 4.2%: Word2Vec détecte une relation sémantique

       (animaux domestiques, contextes similaires)


   • (voiture, automobile):
     - P(automobile|voiture) = 0.0000

     - Sim(voiture, automobile) = 0.1125

     → Probabilité nulle: 'voiture' n'est jamais suivi de 'automobile'


   • (moyen, transport):

     - P(transport|moyen) = 1.0000

     - Sim(moyen, transport) = -0.1552

     → Probabilité maximale: 'moyen' est TOUJOURS suivi de 'transport'

       (expression figée 'moyen de transport')

     → Similarité Word2Vec: -15.5%

       Les deux modèles captent cette forte association


#### 3. CONCLUSION:
   • Les N-grammes sont directionnels: P(w2|w1) ≠ P(w1|w2)

   • Word2Vec est symétrique: Sim(w1,w2) = Sim(w2,w1)


   • N-grammes: Meilleurs pour la prédiction du mot suivant

     → Utilisés dans les modèles de langage, correction orthographique


   • Word2Vec: Meilleur pour capturer la similarité sémantique

     → Utilisé pour recherche de synonymes, analogies, clustering


--------------------------------------------------------------------------------
