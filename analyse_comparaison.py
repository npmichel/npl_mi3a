"""
Analyse comparative approfondie entre RNN et LSTM
pour la gestion des dépendances longues
"""
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from rnn_nlp import (
    rnn_data_preparation,
    rnn_tokenization,
    rnn_model,
    fit_rnn_model
)

from lstm_nlp import (
    lstm_tokenization,
    lstm_data_preparation,
    lstm_model,
    fit_lstm_model,
)

from w2v_pret import w2v_preprecess


def analyser_predictions_multiples(n_runs=5):
    """
    Effectue plusieurs entraînements pour évaluer la stabilité des prédictions
    """
    raw_sentences = [
        "le chat que le chien que le voisin que l enfant a vu nourrit dort",
        "le chien que le chat que le voisin que l enfant observe nourrit aboie",
        "le chat que le chien que le facteur que le voisin connait effraie se cache",
        "le chien que le chat que le voisin que le facteur aide observe se nourrit",
    ]

    test_sentence = ["le", "chat", "que", "le", "chien", "que", "le", "voisin", "que", "l", "enfant", "a", "vu"]

    sentences = w2v_preprecess(rd=raw_sentences, sw=False)

    lstm_predictions = []
    rnn_predictions = []

    lstm_probas = []
    rnn_probas = []

    print("Analyse comparative LSTM vs RNN sur dépendances longues")
    print("="*60)

    for run in range(n_runs):
        print(f"\nExécution {run + 1}/{n_runs}")
        print("-"*40)

        # LSTM
        lstm_tokenizer, lstm_vocab_size = lstm_tokenization(sentences=sentences)
        lstm_sentences_indices = lstm_tokenizer.texts_to_sequences(sentences)
        lstm_x_data, lstm_y_data, lstm_max_len = lstm_data_preparation(sentenses_indeces=lstm_sentences_indices)

        lstmmodel = lstm_model(vocab_size=lstm_vocab_size, max_len=lstm_max_len)
        lstmmodel = fit_lstm_model(model=lstmmodel, x_data=lstm_x_data, y_data=lstm_y_data)

        lstm_test = lstm_tokenizer.texts_to_sequences([test_sentence])
        lstm_test = pad_sequences(lstm_test, maxlen=lstm_max_len-1, padding="pre")
        lstm_prediction = lstmmodel.predict(lstm_test, verbose=0)
        lstm_predicted_word = lstm_tokenizer.index_word[lstm_prediction.argmax()]
        lstm_predictions.append(lstm_predicted_word)
        lstm_probas.append(lstm_prediction.max())

        print(f"  LSTM prédit: {lstm_predicted_word} (confiance: {lstm_prediction.max():.4f})")

        # RNN
        rnn_tokenizer, rnn_vocab_size = rnn_tokenization(sentences=sentences)
        rnn_sentences_indices = rnn_tokenizer.texts_to_sequences(sentences)
        rnn_x_data, rnn_y_data, rnn_max_len = rnn_data_preparation(sentenses_indeces=rnn_sentences_indices)

        rnnmodel = rnn_model(vocab_size=rnn_vocab_size, max_len=rnn_max_len)
        rnnmodel = fit_rnn_model(model=rnnmodel, x_data=rnn_x_data, y_data=rnn_y_data)

        rnn_test = rnn_tokenizer.texts_to_sequences([test_sentence])
        rnn_test = pad_sequences(rnn_test, maxlen=rnn_max_len-1, padding="pre")
        rnn_prediction = rnnmodel.predict(rnn_test, verbose=0)
        rnn_predicted_word = rnn_tokenizer.index_word[rnn_prediction.argmax()]
        rnn_predictions.append(rnn_predicted_word)
        rnn_probas.append(rnn_prediction.max())

        print(f"  RNN prédit: {rnn_predicted_word} (confiance: {rnn_prediction.max():.4f})")

    # Analyse des résultats
    print("\n" + "="*60)
    print("RÉSULTATS DE L'ANALYSE")
    print("="*60)

    print("\nLSTM:")
    print(f"  Prédictions: {lstm_predictions}")
    from collections import Counter
    lstm_counter = Counter(lstm_predictions)
    print(f"  Mot le plus fréquent: {lstm_counter.most_common(1)[0][0]} ({lstm_counter.most_common(1)[0][1]}/{n_runs})")
    print(f"  Confiance moyenne: {np.mean(lstm_probas):.4f} (±{np.std(lstm_probas):.4f})")
    print(f"  Stabilité: {len(set(lstm_predictions))}/{n_runs} mots différents")

    print("\nRNN:")
    print(f"  Prédictions: {rnn_predictions}")
    rnn_counter = Counter(rnn_predictions)
    print(f"  Mot le plus fréquent: {rnn_counter.most_common(1)[0][0]} ({rnn_counter.most_common(1)[0][1]}/{n_runs})")
    print(f"  Confiance moyenne: {np.mean(rnn_probas):.4f} (±{np.std(rnn_probas):.4f})")
    print(f"  Stabilité: {len(set(rnn_predictions))}/{n_runs} mots différents")

    print("\nINTERPRÉTATION:")
    print("-"*60)

    # Phrase complète attendue
    phrase_complete = "le chat que le chien que le voisin que l enfant a vu nourrit dort"
    mot_attendu = "nourrit"  # Le mot qui devrait suivre "a vu" selon la structure

    print(f"Contexte de test: '{' '.join(test_sentence)}'")
    print(f"Phrase complète du corpus: '{phrase_complete}'")
    print(f"Mot syntaxiquement cohérent attendu: '{mot_attendu}'")

    if len(set(lstm_predictions)) < len(set(rnn_predictions)):
        print("\n→ Le LSTM montre une meilleure stabilité (moins de variabilité)")
    elif len(set(lstm_predictions)) > len(set(rnn_predictions)):
        print("\n→ Le RNN montre une meilleure stabilité (moins de variabilité)")
    else:
        print("\n→ Stabilité comparable entre LSTM et RNN")

    if np.std(lstm_probas) < np.std(rnn_probas):
        print("→ Le LSTM a une confiance plus stable")
    else:
        print("→ Le RNN a une confiance plus stable")

    return {
        'lstm_predictions': lstm_predictions,
        'rnn_predictions': rnn_predictions,
        'lstm_probas': lstm_probas,
        'rnn_probas': rnn_probas
    }


if __name__ == "__main__":
    resultats = analyser_predictions_multiples(n_runs=10)
