"""
Module de Prétraitement NLP pour le Français
=============================================

Ce module implémente les fonctions de prétraitement de texte en français
dans le cadre du TP1 d'Intelligence Artificielle - Traitement du Langage Naturel (NLP).

Fonctionnalités:
----------------
1. Normalisation de texte (nettoyage, conversion en minuscules)
2. Tokenisation (NLTK et spaCy)
3. Suppression des stopwords (NLTK et spaCy)
4. Lemmatisation (spaCy)
5. Calcul de fréquences des mots
6. Reconnaissance d'entités nommées (NER)
7. Pipeline de traitement complet
8. Filtrage de ponctuation
9. Comparaison Stemming vs Lemmatisation
10. Extraction de n-grams (bi-grams, tri-grams, etc.)
11. Calcul de longueur moyenne des mots
12. Analyse de la taille du vocabulaire
13. Analyse de fréquences absolues et relatives

Corpus de test:
--------------
Texte sur la Coupe d'Afrique des Nations (CAN) 2025 au Maroc

Auteur: MI3A - TP NLP
Date: 2025
Bibliothèques: NLTK, spaCy (fr_core_news_sm)
"""

import re
import spacy
import nltk
import string


def normalizing(text: str):
    """
    Normalizing text
    """
    print(f"\n Texte original =>: \n \n {text} \n \n ")
    lower_text = text.lower()

    # Remove empty space
    no_space_text = re.sub(r"[^a-zàâçéèêëîïôûù\s]", " ", lower_text)

    #Remove numeric
    no_num_text = re . sub (r"\d+", " ", no_space_text)
    # Remove space
    normalized = re.sub(r"\s+", " ", no_num_text).strip() 

    print(f"\n \n Après normalisation =>: \n \n {normalized} \n \n")

    return normalized



def ntk_tokenization(text:str) -> list[str]:
    """ Tokenization with ntk functions"""
    nltk.download("punkt_tab")
    tokenized = nltk.tokenize.word_tokenize(text=text, language="french")

    print(f"Tokens length is =>: {len(tokenized)}")
    print(f"NTK Les tokens sont =>: \n {tokenized} \n \n ")
    return tokenized


def spacy_tokenization(text:str) -> list[str]:
    """ Tokenization with spacy function"""
    nlp = spacy.load("fr_core_news_sm")
    doc = nlp(text=text)
    tokenized = [token.text for token in doc]

    print(f"Tokens length is =>: {len(tokenized)}")
    print(f"Spacy Les tokens sont =>: \n {tokenized} \n \n ")

    return tokenized


def nltk_stopwords_removal(tokens: list[str]) -> list[str]:
    """Remove stopwords using NLTK"""
    from nltk.corpus import stopwords

    nltk.download("stopwords", quiet=True)
    french_stopwords = set(stopwords.words("french"))

    filtered_tokens = [token for token in tokens if token.lower() not in french_stopwords]

    print(f"\n Nombre de tokens avant suppression des stopwords =>: {len(tokens)}")
    print(f"Nombre de tokens après suppression des stopwords =>: {len(filtered_tokens)}")
    print(f"NLTK Tokens sans stopwords =>: \n {filtered_tokens} \n \n")

    return filtered_tokens


def spacy_stopwords_removal(text: str) -> list[str]:
    """Remove stopwords using spaCy"""
    nlp = spacy.load("fr_core_news_sm")
    doc = nlp(text=text)

    filtered_tokens = [token.text for token in doc if not token.is_stop]

    print(f"\n Nombre de tokens avant suppression des stopwords =>: {len(doc)}")
    print(f"Nombre de tokens après suppression des stopwords =>: {len(filtered_tokens)}")
    print(f"spaCy Tokens sans stopwords =>: \n {filtered_tokens} \n \n")

    return filtered_tokens


def spacy_lemmatization(text: str) -> list[str]:
    """Lemmatization using spaCy"""
    nlp = spacy.load("fr_core_news_sm")
    doc = nlp(text=text)

    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    print(f"\n Tokens originaux =>: {[token.text for token in doc if not token.is_stop and not token.is_punct]}")
    print(f"Lemmes obtenus =>: {lemmas}")
    print(f"Nombre de lemmes =>: {len(lemmas)} \n \n")

    return lemmas


def word_frequencies(tokens: list[str], top_n: int = 10) -> dict[str, int]:
    """Calculate word frequencies from a list of tokens"""
    from collections import Counter

    freq_dict = Counter(tokens)
    most_common = freq_dict.most_common(top_n)

    print(f"\n Nombre total de tokens =>: {len(tokens)}")
    print(f"Nombre de mots uniques =>: {len(freq_dict)}")
    print(f"\n Top {top_n} mots les plus fréquents =>:")
    for word, count in most_common:
        print(f"  - '{word}': {count} occurrences")
    print("\n")

    return dict(freq_dict)


def named_entity_recognition(text: str) -> list[tuple[str, str]]:
    """Named Entity Recognition using spaCy"""
    nlp = spacy.load("fr_core_news_sm")
    doc = nlp(text=text)

    entities = [(ent.text, ent.label_) for ent in doc.ents]

    print(f"\n Nombre d'entités nommées trouvées =>: {len(entities)}")
    print("\n Entités nommées détectées =>:")
    for entity_text, entity_label in entities:
        print(f"  - '{entity_text}' => Type: {entity_label}")
    print("\n")

    return entities


def preprocessing_pipeline(text: str) -> dict:
    """
    Combined preprocessing pipeline
    Returns a dictionary with all preprocessing results
    """
    print("\n" + "="*70)
    print("PIPELINE DE TRAITEMENT COMPLET DU TEXTE")
    print("="*70)

    # Step 1: Normalization
    print("\n[ÉTAPE 1] Normalisation du texte...")
    normalized = normalizing(text)

    # Step 2: Tokenization with spaCy
    print("\n[ÉTAPE 2] Tokenisation avec spaCy...")
    tokens = spacy_tokenization(normalized)

    # Step 3: Stopwords removal
    print("\n[ÉTAPE 3] Suppression des stopwords avec NLTK...")
    filtered_tokens = nltk_stopwords_removal(tokens)

    # Step 4: Lemmatization
    print("\n[ÉTAPE 4] Lemmatisation...")
    lemmas = spacy_lemmatization(normalized)

    # Step 5: Word frequencies
    print("\n[ÉTAPE 5] Calcul des fréquences...")
    frequencies = word_frequencies(lemmas, top_n=15)

    # Step 6: Named Entity Recognition
    print("\n[ÉTAPE 6] Reconnaissance d'entités nommées...")
    entities = named_entity_recognition(text)

    print("\n" + "="*70)
    print("PIPELINE TERMINÉ")
    print("="*70 + "\n")

    return {
        "normalized": normalized,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "lemmas": lemmas,
        "frequencies": frequencies,
        "entities": entities
    }


def remove_punctuation(tokens: list[str]) -> list[str]:
    """Remove punctuation from token list"""
    import string

    punctuation_set = set(string.punctuation)
    filtered = [token for token in tokens if token not in punctuation_set]

    print(f"\n Tokens avant suppression de ponctuation =>: {len(tokens)}")
    print(f"Tokens après suppression de ponctuation =>: {len(filtered)}")
    print(f"Ponctuation retirée =>: {[t for t in tokens if t in punctuation_set]}")
    print(f"Résultat =>: {filtered} \n \n")

    return filtered


def stemming_vs_lemmatization(text: str) -> dict:
    """Compare stemming (NLTK) with lemmatization (spaCy)"""
    from nltk.stem.snowball import SnowballStemmer

    nltk.download("punkt_tab", quiet=True)

    # Tokenization
    tokens = nltk.tokenize.word_tokenize(text=text, language="french")
    tokens_clean = [t.lower() for t in tokens if t.isalpha()]

    # Stemming avec NLTK
    stemmer = SnowballStemmer(language="french")
    stems = [stemmer.stem(token) for token in tokens_clean]

    # Lemmatization avec spaCy
    nlp = spacy.load("fr_core_news_sm")
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha]

    print("\n" + "="*70)
    print("COMPARAISON STEMMING vs LEMMATISATION")
    print("="*70)
    print(f"\n Nombre de tokens =>: {len(tokens_clean)}")
    print(f"\n Exemples de STEMMING (racines) =>:")
    for token, stem in list(zip(tokens_clean, stems))[:10]:
        print(f"  {token:20s} -> {stem}")

    print(f"\n Exemples de LEMMATISATION (lemmes) =>:")
    sample_lemmas = [(token.text, token.lemma_) for token in doc if token.is_alpha][:10]
    for token, lemma in sample_lemmas:
        print(f"  {token:20s} -> {lemma}")

    print("\n DIFFÉRENCES CLÉS =>:")
    print("  - Stemming: coupe les suffixes (plus rapide, moins précis)")
    print("  - Lemmatisation: utilise dictionnaire (plus lent, plus précis)")
    print("="*70 + "\n")

    return {
        "tokens": tokens_clean,
        "stems": stems,
        "lemmas": lemmas
    }


def extract_ngrams(tokens: list[str], n: int = 2) -> list[tuple]:
    """Extract n-grams from a list of tokens"""
    from nltk import ngrams

    ngrams_list = list(ngrams(tokens, n))

    print(f"\n Extraction de {n}-grams")
    print(f"Nombre de tokens =>: {len(tokens)}")
    print(f"Nombre de {n}-grams générés =>: {len(ngrams_list)}")
    print(f"\n Exemples de {n}-grams =>:")

    for i, ngram in enumerate(ngrams_list[:15]):
        print(f"  {i+1}. {' '.join(ngram)}")

    print("\n")

    return ngrams_list


def average_word_length(tokens: list[str]) -> float:
    """Calculate average word length from tokens"""
    if not tokens:
        return 0.0

    total_length = sum(len(token) for token in tokens)
    avg_length = total_length / len(tokens)

    print(f"\n Calcul de la longueur moyenne des mots")
    print(f"Nombre de tokens =>: {len(tokens)}")
    print(f"Longueur totale des caractères =>: {total_length}")
    print(f"Longueur moyenne =>: {avg_length:.2f} caractères")
    print(f"\n Exemples de longueurs =>:")

    for token in tokens[:10]:
        print(f"  '{token}' => {len(token)} caractères")

    print("\n")

    return avg_length


def vocabulary_size(tokens: list[str]) -> dict:
    """Calculate vocabulary statistics"""
    unique_tokens = set(tokens)
    vocab_size = len(unique_tokens)
    total_tokens = len(tokens)

    print(f"\n Analyse de la taille du vocabulaire")
    print(f"Nombre total de tokens =>: {total_tokens}")
    print(f"Nombre de mots uniques (vocabulaire) =>: {vocab_size}")
    print(f"Ratio vocabulaire/total =>: {vocab_size/total_tokens:.2%}")
    print(f"\n Les mots uniques sont =>:")

    sorted_vocab = sorted(unique_tokens)[:20]
    for word in sorted_vocab:
        print(f"  - {word}")

    if len(unique_tokens) > 20:
        print(f"  ... et {len(unique_tokens) - 20} autres mots")

    print("\n")

    return {
        "total_tokens": total_tokens,
        "unique_tokens": vocab_size,
        "ratio": vocab_size / total_tokens
    }


def frequency_analysis(tokens: list[str]) -> dict:
    """Calculate both absolute and relative frequencies"""
    from collections import Counter

    freq_counter = Counter(tokens)
    total_tokens = len(tokens)

    # Absolute frequencies
    absolute_freq = dict(freq_counter.most_common(10))

    # Relative frequencies (proportions)
    relative_freq = {word: count/total_tokens for word, count in absolute_freq.items()}

    print(f"\n Analyse des fréquences (absolues vs relatives)")
    print(f"Nombre total de tokens =>: {total_tokens}")
    print("\n Top 10 mots =>:")
    print(f"{'Mot':<20} {'Fréq. Absolue':<20} {'Fréq. Relative'}")
    print("-" * 60)

    for word in absolute_freq.keys():
        abs_freq = absolute_freq[word]
        rel_freq = relative_freq[word]
        print(f"{word:<20} {abs_freq:<20} {rel_freq:.4f} ({rel_freq*100:.2f}%)")

    print("\n")

    return {
        "absolute_frequencies": absolute_freq,
        "relative_frequencies": relative_freq,
        "total_tokens": total_tokens
    }









    
# ==============================================================================
# DÉMONSTRATION ET TESTS
# ==============================================================================

if __name__ == "__main__":
    """
    Démonstration de toutes les fonctions de prétraitement NLP
    Ce bloc s'exécute quand on lance: python pre_processing.py
    """

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
    normalized = normalizing(text)

    print("\n[TEST 2] Tokenisation avec NLTK")
    print("-" * 80)
    nltk_tokens = ntk_tokenization(normalized)

    print("\n[TEST 2 bis] Tokenisation avec spaCy")
    print("-" * 80)
    spacy_tokens = spacy_tokenization(normalized)

    # Question 3: Suppression des stopwords
    print("\n[TEST 3] Suppression des stopwords avec NLTK")
    print("-" * 80)
    filtered_nltk = nltk_stopwords_removal(nltk_tokens)

    print("\n[TEST 3 bis] Suppression des stopwords avec spaCy")
    print("-" * 80)
    filtered_spacy = spacy_stopwords_removal(normalized)

    # Question 4: Lemmatisation
    print("\n[TEST 4] Lemmatisation avec spaCy")
    print("-" * 80)
    lemmas = spacy_lemmatization(normalized)

    # Question 5: Fréquences des mots
    print("\n[TEST 5] Calcul des fréquences des mots")
    print("-" * 80)
    frequencies = word_frequencies(lemmas, top_n=10)

    # Question 6: Reconnaissance d'entités nommées
    print("\n[TEST 6] Reconnaissance d'entités nommées (NER)")
    print("-" * 80)
    entities = named_entity_recognition(text)

    # Question 8: Pipeline complet
    print("\n[TEST 8] Pipeline de traitement complet")
    print("-" * 80)
    pipeline_result = preprocessing_pipeline(text)

    # Question 9: Suppression de ponctuation
    print("\n[TEST 9] Suppression de la ponctuation")
    print("-" * 80)
    no_punct = remove_punctuation(spacy_tokens[:20])

    # Question 10: Stemming vs Lemmatisation
    print("\n[TEST 10] Comparaison Stemming vs Lemmatisation")
    print("-" * 80)
    stem_vs_lem = stemming_vs_lemmatization(normalized)

    # Question 11: Extraction de n-grams
    print("\n[TEST 11] Extraction de bi-grams (n=2)")
    print("-" * 80)
    bigrams = extract_ngrams(filtered_nltk[:30], n=2)

    print("\n[TEST 11 bis] Extraction de tri-grams (n=3)")
    print("-" * 80)
    trigrams = extract_ngrams(filtered_nltk[:30], n=3)

    # Question 12: Longueur moyenne des mots
    print("\n[TEST 12] Calcul de la longueur moyenne des mots")
    print("-" * 80)
    avg_length = average_word_length(filtered_nltk)

    # Question 13: Taille du vocabulaire
    print("\n[TEST 13] Analyse de la taille du vocabulaire")
    print("-" * 80)
    vocab_stats = vocabulary_size(lemmas)

    # Question 14: Fréquences absolues vs relatives
    print("\n[TEST 14] Analyse des fréquences absolues vs relatives")
    print("-" * 80)
    freq_analysis = frequency_analysis(lemmas)

    print("\n" + "="*80)
    print(" FIN DE LA DÉMONSTRATION ".center(80, "="))
    print("="*80 + "\n")

    print("\n" + "="*80)
    print(" RÉSUMÉ DES RÉSULTATS ".center(80, "="))
    print("="*80)
    print(f"\n  📊 Statistiques générales:")
    print(f"    • Texte normalisé: {len(normalized)} caractères")
    print(f"    • Tokens NLTK: {len(nltk_tokens)}")
    print(f"    • Tokens spaCy: {len(spacy_tokens)}")
    print(f"    • Tokens après stopwords (NLTK): {len(filtered_nltk)}")
    print(f"    • Lemmes: {len(lemmas)}")

    print(f"\n  🏷️  Entités nommées:")
    print(f"    • Nombre d'entités détectées: {len(entities)}")

    print(f"\n  📖 Analyse du vocabulaire:")
    print(f"    • Mots uniques: {vocab_stats['unique_tokens']}")
    print(f"    • Longueur moyenne: {avg_length:.2f} caractères/mot")
    print(f"    • Ratio vocabulaire/total: {vocab_stats['ratio']:.2%}")

    print(f"\n  🔤 N-grams:")
    print(f"    • Bi-grams générés: {len(bigrams)}")
    print(f"    • Tri-grams générés: {len(trigrams)}")

    print("\n" + "="*80)
    print(" ✅ TOUTES LES FONCTIONS ONT ÉTÉ TESTÉES AVEC SUCCÈS ".center(80, "="))