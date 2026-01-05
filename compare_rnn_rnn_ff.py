import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


from w2v_pret import w2v_preprecess
from rnn_feed_foward import ( 
    rnn_ff_tokenization, fit_rnn_ff_model, 
    nn_feed_foward_model, rnn_ff_data_preparation
)

from rnn_nlp import (
    rnn_data_preparation,
    rnn_tokenization,
    rnn_model,
    fit_rnn_model
)



raw_sentenses = [
        "le chat que le chien a vu mange",
        "le chien que le chat poursuit aboie",
        "le chat que le voisin nourrit dort",
        "le chien que le voisin adopte mange",
        "le chat que le chien effraie se cache",
        "le chien que le chat observe se nourrit",
    ]


sentenses = w2v_preprecess(rd=raw_sentenses, sw=False)


# FF
ffN = 6
print(f"FF")
ff_tokenizer, ff_vocab_size = rnn_ff_tokenization(sentences=sentenses)

#2. Data 
ff_x_data, ff_y_data = rnn_ff_data_preparation(sentenses=sentenses, tokenizer=ff_tokenizer, context_size=ffN)

# 3. Model 
ff_model = nn_feed_foward_model(vocab_size=ff_vocab_size, context_size=ffN)

# 4. Training
ff_model = fit_rnn_ff_model(model=ff_model, x_data=ff_x_data, y_data=ff_y_data)

# Question 6
ff_2_test = ff_tokenizer.texts_to_sequences([["le", "chat", "que", "le", "chien", "a"]]) # Convertion du contexte en indices
ff_prediction = ff_model.predict(np.array(ff_2_test)) # Prediction des probabilités
predicted_word = ff_tokenizer.index_word[ff_prediction.argmax()] # Mot avec probabilité maximale
print(f"FF-2 Mot prédit par  =>: {predicted_word}")


print(f" RNN")
# 1. Tokenizer
rnn_tokenizer, rnn_vocab_size = rnn_tokenization(sentences=sentenses)
setences_indices = rnn_tokenizer.texts_to_sequences(sentenses)

#2. Data 
rnn_x_data, rnn_y_data, max_len = rnn_data_preparation (sentenses_indeces=setences_indices)

# 3. Model 
rnn_model = rnn_model(vocab_size=rnn_vocab_size, max_len=max_len)

# 4. Training
rnn_model = fit_rnn_model(model=rnn_model, x_data=rnn_x_data, y_data=rnn_y_data)

# Prediction
rnn_test_sentence = ["le", "chat", "que", "le", "chien", "a", "vu"]
# Conversion en indices
rnn_test = rnn_tokenizer.texts_to_sequences([rnn_test_sentence])

# Padding pour correspondre au format d'entree du model
rnn_test = pad_sequences(rnn_test, maxlen=max_len-1, padding="pre")

# Prediction
rnn_prediction = rnn_model.predict(rnn_test)
rnn_predicted_word = rnn_tokenizer.index_word[rnn_prediction.argmax()]
top_indices = rnn_prediction[0].argsort()[-3:][::-1]
top_words = [rnn_tokenizer.index_word[i] for i in top_indices]
print(f"RNN Vector =>: {top_words}")
print(f"Mot predit par RNN =>: {rnn_predicted_word}")
