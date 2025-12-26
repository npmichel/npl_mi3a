"""
Démonstration des fonctions de prétraitement NLP
Ce fichier teste toutes les fonctions implémentées dans pre_processing.py
"""
import pre_processing


def main():
    """Fonction principale de démonstration"""

    # Texte d'exemple (CAN 2025)
    text = """
    La Coupe d'Afrique des Nations 2025 (CAN 2025) se tiendra au Maroc ! C'est
    une compétition très attendue par les fans de football. L'événement débutera officiellement en décembre 2025, avec des matchs dans plusieurs villes :
    Casablanca, Rabat, Marrakech, Fès et Tanger. Les organisateurs annoncent déjà plus de 100.000 billets vendus en ligne.
    Le site officiel (https ://www.cafonline.com)
    propose toutes les informations nécessaires. Contact presse : info@can2025.ma
    Cependant, certains supporters s'inquiètent du prix des tickets, jugés parfois «
    trop élevés ». Le Maroc, qui avait déjà accueilli de grands événements sportifs
    internationaux, espère renforcer son image et montrer sa capacité d'organisation.
    Les Lions de l'Atlas, quant à eux, rêvent de soulever le trophée devant leur public !
    """

    print("\n" + "="*80)
    print(" DÉMONSTRATION DES FONCTIONS DE PRÉTRAITEMENT NLP ".center(80, "="))
    print("="*80 + "\n")

    # Question 1 & 2: Normalisation et Tokenisation
    print("\n[TEST 1] Normalisation du texte")
    print("-" * 80)
    normalized = pre_processing.normalizing(text)

    print("\n[TEST 2] Tokenisation avec NLTK")
    print("-" * 80)
    nltk_tokens = pre_processing.ntk_tokenization(normalized)

    print("\n[TEST 2 bis] Tokenisation avec spaCy")
    print("-" * 80)
    spacy_tokens = pre_processing.spacy_tokenization(normalized)

    # Question 3: Suppression des stopwords
    print("\n[TEST 3] Suppression des stopwords avec NLTK")
    print("-" * 80)
    filtered_nltk = pre_processing.nltk_stopwords_removal(nltk_tokens)

    print("\n[TEST 3 bis] Suppression des stopwords avec spaCy")
    print("-" * 80)
    filtered_spacy = pre_processing.spacy_stopwords_removal(normalized)

    # Question 4: Lemmatisation
    print("\n[TEST 4] Lemmatisation avec spaCy")
    print("-" * 80)
    lemmas = pre_processing.spacy_lemmatization(normalized)

    # Question 5: Fréquences des mots
    print("\n[TEST 5] Calcul des fréquences des mots")
    print("-" * 80)
    frequencies = pre_processing.word_frequencies(lemmas, top_n=10)

    # Question 6: Reconnaissance d'entités nommées
    print("\n[TEST 6] Reconnaissance d'entités nommées (NER)")
    print("-" * 80)
    entities = pre_processing.named_entity_recognition(text)

    # Question 8: Pipeline complet
    print("\n[TEST 8] Pipeline de traitement complet")
    print("-" * 80)
    pipeline_result = pre_processing.preprocessing_pipeline(text)

    # Question 9: Suppression de ponctuation
    print("\n[TEST 9] Suppression de la ponctuation")
    print("-" * 80)
    no_punct = pre_processing.remove_punctuation(spacy_tokens[:20])

    # Question 10: Stemming vs Lemmatisation
    print("\n[TEST 10] Comparaison Stemming vs Lemmatisation")
    print("-" * 80)
    stem_vs_lem = pre_processing.stemming_vs_lemmatization(normalized)

    # Question 11: Extraction de n-grams
    print("\n[TEST 11] Extraction de bi-grams (n=2)")
    print("-" * 80)
    bigrams = pre_processing.extract_ngrams(filtered_nltk[:30], n=2)

    print("\n[TEST 11 bis] Extraction de tri-grams (n=3)")
    print("-" * 80)
    trigrams = pre_processing.extract_ngrams(filtered_nltk[:30], n=3)

    # Question 12: Longueur moyenne des mots
    print("\n[TEST 12] Calcul de la longueur moyenne des mots")
    print("-" * 80)
    avg_length = pre_processing.average_word_length(filtered_nltk)

    # Question 13: Taille du vocabulaire
    print("\n[TEST 13] Analyse de la taille du vocabulaire")
    print("-" * 80)
    vocab_stats = pre_processing.vocabulary_size(lemmas)

    # Question 14: Fréquences absolues vs relatives
    print("\n[TEST 14] Analyse des fréquences absolues vs relatives")
    print("-" * 80)
    freq_analysis = pre_processing.frequency_analysis(lemmas)

    print("\n" + "="*80)
    print(" FIN DE LA DÉMONSTRATION ".center(80, "="))
    print("="*80 + "\n")

    print("\nRésumé des résultats:")
    print(f"  - Texte normalisé: {len(normalized)} caractères")
    print(f"  - Tokens NLTK: {len(nltk_tokens)}")
    print(f"  - Tokens spaCy: {len(spacy_tokens)}")
    print(f"  - Lemmes: {len(lemmas)}")
    print(f"  - Entités nommées: {len(entities)}")
    print(f"  - Vocabulaire unique: {vocab_stats['unique_tokens']} mots")
    print(f"  - Longueur moyenne: {avg_length:.2f} caractères/mot")
    print(f"  - Bi-grams: {len(bigrams)}")
    print(f"  - Tri-grams: {len(trigrams)}")


if __name__ == "__main__":
    main()
