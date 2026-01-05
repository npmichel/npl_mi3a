from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


from rnn_nlp import (
    rnn_data_preparation,
    rnn_tokenization,
    rnn_model,fit_rnn_model
    )

from lstm_nlp import (
    lstm_tokenization,
    lstm_data_preparation,
    lstm_model,
    fit_lstm_model,
)

from w2v_pret import w2v_preprecess


raw_sentenses = [
    "le chat que le chien que le voisin que l enfant a vu nourrit dort",
    "le chien que le chat que le voisin que l enfant observe nourrit aboie",
    "le chat que le chien que le facteur que le voisin connait effraie se cache",
    "le chien que le chat que le voisin que le facteur aide observe se nourrit",
]
test_sentence = ["le", "chat", "que", "le", "chien", "que", "le", "voisin", "que", "l", "enfant", "a", "vu"]

sentenses = w2v_preprecess(rd=raw_sentenses, sw=False)
print(f"Le sentense est =>: {sentenses}")
print(f" LTM")
# 1. Tokenizer
lstm_tokenizer, lstm_vocab_size = lstm_tokenization(sentences=sentenses)
lstm_setences_indices = lstm_tokenizer.texts_to_sequences(sentenses)

#2. Data 
lstm_x_data, lstm_y_data, lstm_max_len = lstm_data_preparation (sentenses_indeces=lstm_setences_indices)

# 3. Model 
lstmmodel = lstm_model(vocab_size=lstm_vocab_size, max_len=lstm_max_len)

# 4. Training
lstmmodel = fit_lstm_model(model=lstmmodel, x_data=lstm_x_data, y_data=lstm_y_data)

# Prediction
lstm_test_sentence = test_sentence
# Conversion en indices
lstm_test = lstm_tokenizer.texts_to_sequences([lstm_test_sentence])

# Padding pour correspondre au format d'entree du model
lstm_test = pad_sequences(lstm_test, maxlen=lstm_max_len-1, padding="pre")

# Prediction
lstm_prediction = lstmmodel.predict(lstm_test)
lstm_predicted_word = lstm_tokenizer.index_word[lstm_prediction.argmax()]
print(f"Mot predit par LSTM =>: {lstm_predicted_word}")

    


print(f" RNN")
# 1. Tokenizer
rnn_tokenizer, rnn_vocab_size = rnn_tokenization(sentences=sentenses)
rnn_setences_indices = rnn_tokenizer.texts_to_sequences(sentenses)

#2. Data 
rnn_x_data, rnn_y_data, rnn_max_len = rnn_data_preparation (sentenses_indeces=rnn_setences_indices)

# 3. Model 
rnnmodel = rnn_model(vocab_size=rnn_vocab_size, max_len=rnn_max_len)

# 4. Training
rnnmodel = fit_rnn_model(model=rnnmodel, x_data=rnn_x_data, y_data=rnn_y_data)

# Prediction
rnn_test_sentence = test_sentence
# Conversion en indices
rnn_test = rnn_tokenizer.texts_to_sequences([rnn_test_sentence])

# Padding pour correspondre au format d'entree du model
rnn_test = pad_sequences(rnn_test, maxlen=rnn_max_len-1, padding="pre")

# Prediction
rnn_prediction = rnnmodel.predict(rnn_test)
rnn_predicted_word = rnn_tokenizer.index_word[rnn_prediction.argmax()]
print(f"Mot predit par RNN =>: {rnn_predicted_word}")
