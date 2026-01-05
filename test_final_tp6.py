"""
Script de test final pour le TP6 - LSTM vs RNN
Génère les résultats principaux pour le rapport
"""
from tensorflow.keras.preprocessing.sequence import pad_sequences
from w2v_pret import w2v_preprecess
from lstm_nlp import lstm_tokenization, lstm_data_preparation, lstm_model, fit_lstm_model
from rnn_nlp import rnn_tokenization, rnn_data_preparation, rnn_model, fit_rnn_model


def test_unique_execution():
    """
    Test avec une seule exécution pour chaque modèle
    Résultats à inclure dans le rapport
    """
    print("="*70)
    print("TP6 - TEST FINAL : Comparaison LSTM vs RNN sur dépendances longues")
    print("="*70)

    # Corpus avec dépendances longues
    raw_sentences = [
        "le chat que le chien que le voisin que l enfant a vu nourrit dort",
        "le chien que le chat que le voisin que l enfant observe nourrit aboie",
        "le chat que le chien que le facteur que le voisin connait effraie se cache",
        "le chien que le chat que le voisin que le facteur aide observe se nourrit",
    ]

    sentences = w2v_preprecess(rd=raw_sentences, sw=False)

    print("\n1. CORPUS D'ENTRAÎNEMENT")
    print("-" * 70)
    for i, phrase in enumerate(raw_sentences, 1):
        print(f"   Phrase {i}: {phrase}")

    # Phrase de test
    test_sentence = ["le", "chat", "que", "le", "chien", "que", "le", "voisin", "que", "l", "enfant", "a", "vu"]

    print("\n2. SÉQUENCE DE TEST")
    print("-" * 70)
    print(f"   Contexte: '{' '.join(test_sentence)}'")
    print(f"   Objectif: Prédire le mot suivant")
    print(f"   Phrase complète attendue: '{raw_sentences[0]}'")
    print(f"   → Mot syntaxiquement cohérent: 'nourrit'")

    # TEST LSTM
    print("\n3. TEST DU MODÈLE LSTM")
    print("-" * 70)

    lstm_tokenizer, lstm_vocab_size = lstm_tokenization(sentences=sentences)
    lstm_sentences_indices = lstm_tokenizer.texts_to_sequences(sentences)
    lstm_x_data, lstm_y_data, lstm_max_len = lstm_data_preparation(sentenses_indeces=lstm_sentences_indices)

    print(f"   Architecture:")
    print(f"     - Vocabulaire: {lstm_vocab_size} mots")
    print(f"     - Embedding: dimension 8")
    print(f"     - LSTM: 16 unités mémoire")
    print(f"     - Époques: 300")

    lstmmodel = lstm_model(vocab_size=lstm_vocab_size, max_len=lstm_max_len)
    print(f"   Entraînement en cours...")
    lstmmodel = fit_lstm_model(model=lstmmodel, x_data=lstm_x_data, y_data=lstm_y_data)

    lstm_test = lstm_tokenizer.texts_to_sequences([test_sentence])
    lstm_test = pad_sequences(lstm_test, maxlen=lstm_max_len-1, padding="pre")
    lstm_prediction = lstmmodel.predict(lstm_test, verbose=0)
    lstm_predicted_word = lstm_tokenizer.index_word[lstm_prediction.argmax()]
    lstm_confidence = lstm_prediction.max()

    # Top 3 prédictions LSTM
    lstm_top3_indices = lstm_prediction[0].argsort()[-3:][::-1]
    lstm_top3 = [(lstm_tokenizer.index_word[idx], lstm_prediction[0][idx]) for idx in lstm_top3_indices]

    print(f"\n   Résultat LSTM:")
    print(f"     Mot prédit: '{lstm_predicted_word}' (confiance: {lstm_confidence:.4f})")
    print(f"     Top 3 prédictions:")
    for i, (mot, prob) in enumerate(lstm_top3, 1):
        print(f"       {i}. '{mot}' (probabilité: {prob:.4f})")

    # TEST RNN
    print("\n4. TEST DU MODÈLE RNN")
    print("-" * 70)

    rnn_tokenizer, rnn_vocab_size = rnn_tokenization(sentences=sentences)
    rnn_sentences_indices = rnn_tokenizer.texts_to_sequences(sentences)
    rnn_x_data, rnn_y_data, rnn_max_len = rnn_data_preparation(sentenses_indeces=rnn_sentences_indices)

    print(f"   Architecture:")
    print(f"     - Vocabulaire: {rnn_vocab_size} mots")
    print(f"     - Embedding: dimension 8")
    print(f"     - SimpleRNN: 16 unités récurrentes")
    print(f"     - Dense: 16 unités (ReLU)")
    print(f"     - Époques: 200")

    rnnmodel = rnn_model(vocab_size=rnn_vocab_size, max_len=rnn_max_len)
    print(f"   Entraînement en cours...")
    rnnmodel = fit_rnn_model(model=rnnmodel, x_data=rnn_x_data, y_data=rnn_y_data)

    rnn_test = rnn_tokenizer.texts_to_sequences([test_sentence])
    rnn_test = pad_sequences(rnn_test, maxlen=rnn_max_len-1, padding="pre")
    rnn_prediction = rnnmodel.predict(rnn_test, verbose=0)
    rnn_predicted_word = rnn_tokenizer.index_word[rnn_prediction.argmax()]
    rnn_confidence = rnn_prediction.max()

    # Top 3 prédictions RNN
    rnn_top3_indices = rnn_prediction[0].argsort()[-3:][::-1]
    rnn_top3 = [(rnn_tokenizer.index_word[idx], rnn_prediction[0][idx]) for idx in rnn_top3_indices]

    print(f"\n   Résultat RNN:")
    print(f"     Mot prédit: '{rnn_predicted_word}' (confiance: {rnn_confidence:.4f})")
    print(f"     Top 3 prédictions:")
    for i, (mot, prob) in enumerate(rnn_top3, 1):
        print(f"       {i}. '{mot}' (probabilité: {prob:.4f})")

    # COMPARAISON
    print("\n5. COMPARAISON ET ANALYSE")
    print("-" * 70)

    print(f"   Mot attendu: 'nourrit'")
    print(f"   LSTM prédit: '{lstm_predicted_word}' (confiance: {lstm_confidence:.4f})")
    print(f"   RNN prédit:  '{rnn_predicted_word}' (confiance: {rnn_confidence:.4f})")

    if lstm_predicted_word == "nourrit":
        print(f"\n   ✓ Le LSTM a correctement prédit le mot syntaxiquement cohérent")
    elif lstm_predicted_word in ["dort", "aboie", "cache", "nourrit"]:
        print(f"\n   ~ Le LSTM a prédit un verbe final du corpus")
        if lstm_predicted_word == "dort":
            print(f"     Note: 'dort' est le verbe principal de la phrase complète")
            print(f"           s'accordant avec 'le chat' (sujet initial)")

    if rnn_predicted_word == "nourrit":
        print(f"\n   ✓ Le RNN a correctement prédit le mot syntaxiquement cohérent")
    elif rnn_predicted_word == "aboie":
        print(f"\n   ✗ Le RNN prédit 'aboie' qui s'accorde avec 'le chien'")
        print(f"     Indication d'une perte de l'information sur le sujet initial 'le chat'")

    print("\n   Observations:")
    if lstm_confidence < rnn_confidence:
        print(f"   - Le LSTM montre moins de confiance que le RNN")
        print(f"     Cela peut refléter une meilleure estimation de l'incertitude")

    # Analyse de la distribution des probabilités
    lstm_entropy = -sum([p * np.log(p + 1e-10) for p in lstm_prediction[0] if p > 0])
    rnn_entropy = -sum([p * np.log(p + 1e-10) for p in rnn_prediction[0] if p > 0])

    print(f"   - Entropie LSTM: {lstm_entropy:.4f}")
    print(f"   - Entropie RNN:  {rnn_entropy:.4f}")
    if lstm_entropy > rnn_entropy:
        print(f"     → Le LSTM a une distribution plus uniforme (plus d'incertitude)")
    else:
        print(f"     → Le RNN a une distribution plus uniforme (plus d'incertitude)")

    print("\n" + "="*70)
    print("FIN DU TEST")
    print("="*70)


if __name__ == "__main__":
    import numpy as np
    test_unique_execution()
