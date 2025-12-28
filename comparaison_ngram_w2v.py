"""
Comparaison entre N-grammes et Word2Vec
=========================================

Question 4 du TP: Comparer quantitativement les modèles n-grammes et Word2Vec
sur les mêmes paires de mots.

Paires de mots à analyser:
- (chat, chien)
- (voiture, automobile)
- (moyen, transport)
"""

from collections import Counter
from gensim.models import Word2Vec
from w2v_pret import w2v_preprecess
from ngramme import ngram_process, prob_ngram
from cont_word_to_vector import consine_sim


def compare_ngram_w2v():
    """
    Compare les modèles N-grammes et Word2Vec sur les mêmes paires de mots
    """
    print("COMPARAISON N-GRAMMES vs WORD2VEC ".center(5, "="))
    print("="*5 + "\n")

    # Prétraitement des données
    sentences = w2v_preprecess()

    # ========================================================================
    # PARTIE 1: Calcul des probabilités avec les bigrammes
    # ========================================================================
    print("\n ## [PARTIE 1] Calcul des probabilités conditionnelles avec les bigrammes\n")
    print("-"*80)

    # Générer les bigrammes
    bigrams = ngram_process(sentences=sentences, ng=2)

    # Générer les unigrammes (pour le dénominateur)
    unigrams = Counter([w for sent in sentences for w in sent])

    # Paires de mots à tester
    word_pairs = [
        ("chat", "chien"),
        ("voiture", "automobile"),
        ("moyen", "transport")
    ]

    # Calculer P(w2|w1) pour chaque paire
    bigram_probs = {}
    for w1, w2 in word_pairs:
        prob = prob_ngram(w2=w2, w1=w1, ngf=bigrams, cn=unigrams)
        bigram_probs[(w1, w2)] = prob
        print(f"  P({w2} | {w1}) = {prob:.4f} \n ")

    # ========================================================================
    # PARTIE 2: Calcul de similarité cosinus avec Word2Vec
    # ========================================================================
    print("\n\n ## [PARTIE 2] Calcul de la similarité cosinus avec Word2Vec\n")
    print("-"*80)

    # Entraîner le modèle Word2Vec
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=2,
        min_count=1,
        sg=0  # CBOW
    )

    # Calculer la similarité cosinus pour chaque paire
    w2v_similarities = {}
    for w1, w2 in word_pairs:
        try:
            sim = consine_sim(model.wv[w1], model.wv[w2])
            w2v_similarities[(w1, w2)] = sim
            print(f"  Sim({w1}, {w2}) = {sim:.4f} \n")
        except KeyError as e:
            print(f"  Erreur: {e} n'est pas dans le vocabulaire")
            w2v_similarities[(w1, w2)] = None

    # ========================================================================
    # PARTIE 3: Présentation des résultats sous forme de tableau
    # ========================================================================
    print("\n\n ## [PARTIE 3] Tableau comparatif des résultats\n")
    # print("="*80)
    print(f" ### {'Paire de mots':<25} {'P(w2|w1) Bigrammes':<25} {'Similarité Word2Vec':<25}\n")
    # print("="*80)
    print("-"*80)

    for w1, w2 in word_pairs:
        pair_name = f"({w1}, {w2})"
        prob = bigram_probs[(w1, w2)]
        sim = w2v_similarities[(w1, w2)]

        if sim is not None:
            print(f"{pair_name:<25} {prob:<25.4f} {sim:<25.4f} \n")
        else:
            print(f"{pair_name:<25} {prob:<25.4f} {'N/A':<25}")

    # print("="*80)
    print("-"*80)


    # ========================================================================
    # PARTIE 4: Interprétation des différences observées
    # ========================================================================
    print("\n\n ## [PARTIE 4] Interprétation des différences observées\n")
    # print("="*80)
    print("-"*80)


    print("\n ###  ANALYSE COMPARATIVE:\n")

    print("#### 1. DIFFÉRENCES FONDAMENTALES:")
    print("   • N-grammes: Mesure la probabilité conditionnelle P(w2|w1)\n")
    print("     → Basé sur la fréquence d'occurrence séquentielle dans le corpus\n")
    print("     → Capture l'ordre des mots et leur co-occurrence directe\n")
    print()
    print("   • Word2Vec: Mesure la similarité sémantique via cosinus\n")
    print("     → Basé sur les contextes partagés (fenêtre de mots voisins)\n")
    print("     → Capture les relations sémantiques, pas nécessairement la séquence\n")
    print()

    print("#### 2.   OBSERVATIONS SUR LES RÉSULTATS:")
    print()

    # Analyser (chat, chien)
    prob_chat_chien = bigram_probs[("chat", "chien")]
    sim_chat_chien = w2v_similarities[("chat", "chien")]
    print(f"   • (chat, chien):\n")
    print(f"     - P(chien|chat) = {prob_chat_chien:.4f}\n")
    print(f"     - Sim(chat, chien) = {sim_chat_chien:.4f}\n")
    if prob_chat_chien == 0:
        print("     → Probabilité nulle: 'chat' n'est jamais suivi de 'chien' dans le corpus\n")
    else:
        print(f"     → Probabilité {prob_chat_chien:.1%}: 'chien' suit 'chat' dans certains contextes\n")
    print(f"     → Similarité {sim_chat_chien:.1%}: Word2Vec détecte une relation sémantique\n")
    print("       (animaux domestiques, contextes similaires)\n")
    print()

    # Analyser (voiture, automobile)
    prob_voiture_auto = bigram_probs[("voiture", "automobile")]
    sim_voiture_auto = w2v_similarities[("voiture", "automobile")]
    print(f"   • (voiture, automobile):")
    print(f"     - P(automobile|voiture) = {prob_voiture_auto:.4f}\n")
    print(f"     - Sim(voiture, automobile) = {sim_voiture_auto:.4f}\n")
    if prob_voiture_auto == 0:
        print("     → Probabilité nulle: 'voiture' n'est jamais suivi de 'automobile'\n")
    else:
        print(f"     → Probabilité {prob_voiture_auto:.1%}")
    if sim_voiture_auto and sim_voiture_auto > 0.5:
        print(f"     → Similarité ÉLEVÉE ({sim_voiture_auto:.1%}): Ce sont des synonymes!\n")
        print("       Word2Vec excelle à détecter la synonymie\n")
    print()

    # Analyser (moyen, transport)
    prob_moyen_transport = bigram_probs[("moyen", "transport")]
    sim_moyen_transport = w2v_similarities[("moyen", "transport")]
    print(f"   • (moyen, transport):\n")
    print(f"     - P(transport|moyen) = {prob_moyen_transport:.4f}\n")
    print(f"     - Sim(moyen, transport) = {sim_moyen_transport:.4f}\n")
    if prob_moyen_transport == 1.0:
        print("     → Probabilité maximale: 'moyen' est TOUJOURS suivi de 'transport'\n")
        print("       (expression figée 'moyen de transport')\n")
    print(f"     → Similarité Word2Vec: {sim_moyen_transport:.1%}\n")
    print("       Les deux modèles captent cette forte association\n")
    print()

    print("#### 3. CONCLUSION:")
    print("   • Les N-grammes sont directionnels: P(w2|w1) ≠ P(w1|w2)\n")
    print("   • Word2Vec est symétrique: Sim(w1,w2) = Sim(w2,w1)\n")
    print()
    print("   • N-grammes: Meilleurs pour la prédiction du mot suivant\n")
    print("     → Utilisés dans les modèles de langage, correction orthographique\n")
    print()
    print("   • Word2Vec: Meilleur pour capturer la similarité sémantique\n")
    print("     → Utilisé pour recherche de synonymes, analogies, clustering\n")
    print()

    # print("="*80 + "\n")
    print("-"*80)


    return {
        "bigram_probs": bigram_probs,
        "w2v_similarities": w2v_similarities,
        "word_pairs": word_pairs
    }


if __name__ == "__main__":
    results = compare_ngram_w2v()

    # print("\n✅ Comparaison terminée avec succès!\n")
