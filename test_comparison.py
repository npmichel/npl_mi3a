import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# Corpus
raw_sentenses = [
    "le chat que le chien a vu mange",
    "le chien que le chat poursuit aboie",
    "le chat que le voisin nourrit dort",
    "le chien que le voisin adopte mange",
    "le chat que le chien effraie se cache",
    "le chien que le chat observe se nourrit",
]

sentenses = w2v_preprecess(rd=raw_sentenses, sw=False)
print("Corpus prétraité:", sentenses)
print("\n" + "="*80)

# Test 1: Feed-Forward avec contexte de 2 mots (FF-2)
print("\n1. FEED-FORWARD avec contexte de 2 mots (FF-2)")
print("-"*80)

ff2_tokenizer, ff2_vocab_size = rnn_ff_tokenization(sentences=sentenses)
ff2_x_data, ff2_y_data = rnn_ff_data_preparation(sentenses=sentenses, tokenizer=ff2_tokenizer, context_size=2)
ff2_model = nn_feed_foward_model(vocab_size=ff2_vocab_size, context_size=2)
ff2_model = fit_rnn_ff_model(model=ff2_model, x_data=ff2_x_data, y_data=ff2_y_data)

# Test avec "le chat"
ff2_test = ff2_tokenizer.texts_to_sequences([["le", "chat"]])
ff2_prediction = ff2_model.predict(np.array(ff2_test), verbose=0)
ff2_predicted_word = ff2_tokenizer.index_word[ff2_prediction.argmax()]
ff2_top_indices = ff2_prediction[0].argsort()[-3:][::-1]
ff2_top_words = [ff2_tokenizer.index_word[i] for i in ff2_top_indices]
ff2_top_probs = [ff2_prediction[0][i] for i in ff2_top_indices]

print(f"Contexte: ['le', 'chat']")
print(f"Mot prédit: {ff2_predicted_word}")
print(f"Top 3 prédictions: {ff2_top_words}")
print(f"Probabilités: {[f'{p:.3f}' for p in ff2_top_probs]}")

print("\n" + "="*80)

# Test 2: Feed-Forward avec contexte de 6 mots (FF-6)
print("\n2. FEED-FORWARD avec contexte de 6 mots (FF-6)")
print("-"*80)

ff6_tokenizer, ff6_vocab_size = rnn_ff_tokenization(sentences=sentenses)
ff6_x_data, ff6_y_data = rnn_ff_data_preparation(sentenses=sentenses, tokenizer=ff6_tokenizer, context_size=6)
ff6_model = nn_feed_foward_model(vocab_size=ff6_vocab_size, context_size=6)
ff6_model = fit_rnn_ff_model(model=ff6_model, x_data=ff6_x_data, y_data=ff6_y_data)

# Test avec "le chat que le chien a"
ff6_test = ff6_tokenizer.texts_to_sequences([["le", "chat", "que", "le", "chien", "a"]])
ff6_prediction = ff6_model.predict(np.array(ff6_test), verbose=0)
ff6_predicted_word = ff6_tokenizer.index_word[ff6_prediction.argmax()]
ff6_top_indices = ff6_prediction[0].argsort()[-3:][::-1]
ff6_top_words = [ff6_tokenizer.index_word[i] for i in ff6_top_indices]
ff6_top_probs = [ff6_prediction[0][i] for i in ff6_top_indices]

print(f"Contexte: ['le', 'chat', 'que', 'le', 'chien', 'a']")
print(f"Mot prédit: {ff6_predicted_word}")
print(f"Top 3 prédictions: {ff6_top_words}")
print(f"Probabilités: {[f'{p:.3f}' for p in ff6_top_probs]}")

print("\n" + "="*80)

# Test 3: RNN
print("\n3. RNN (Recurrent Neural Network)")
print("-"*80)

rnn_tokenizer, rnn_vocab_size = rnn_tokenization(sentences=sentenses)
setences_indices = rnn_tokenizer.texts_to_sequences(sentenses)
rnn_x_data, rnn_y_data, max_len = rnn_data_preparation(sentenses_indeces=setences_indices)
rnn_model_obj = rnn_model(vocab_size=rnn_vocab_size, max_len=max_len)
rnn_model_obj = fit_rnn_model(model=rnn_model_obj, x_data=rnn_x_data, y_data=rnn_y_data)

# Test avec "le chat que le chien a vu"
rnn_test_sentence = ["le", "chat", "que", "le", "chien", "a", "vu"]
rnn_test = rnn_tokenizer.texts_to_sequences([rnn_test_sentence])
rnn_test = pad_sequences(rnn_test, maxlen=max_len-1, padding="pre")
rnn_prediction = rnn_model_obj.predict(rnn_test, verbose=0)
rnn_predicted_word = rnn_tokenizer.index_word[rnn_prediction.argmax()]
rnn_top_indices = rnn_prediction[0].argsort()[-3:][::-1]
rnn_top_words = [rnn_tokenizer.index_word[i] for i in rnn_top_indices]
rnn_top_probs = [rnn_prediction[0][i] for i in rnn_top_indices]

print(f"Contexte: {rnn_test_sentence}")
print(f"Mot prédit: {rnn_predicted_word}")
print(f"Top 3 prédictions: {rnn_top_words}")
print(f"Probabilités: {[f'{p:.3f}' for p in rnn_top_probs]}")

print("\n" + "="*80)
print("\nRÉSUMÉ COMPARATIF")
print("="*80)
print(f"\nFF-2 (contexte: 'le chat'):")
print(f"  → Prédit: {ff2_predicted_word}")
print(f"\nFF-6 (contexte: 'le chat que le chien a'):")
print(f"  → Prédit: {ff6_predicted_word}")
print(f"\nRNN (contexte: 'le chat que le chien a vu'):")
print(f"  → Prédit: {rnn_predicted_word}")
print("\n" + "="*80)
