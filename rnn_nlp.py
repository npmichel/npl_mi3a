from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer # pyright: ignore[reportMissingImports] # Tokenization tools
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential # type: ignore # Model Sequentiel
from tensorflow.keras.layers import Embedding, SimpleRNN, Flatten , Dense # type: ignore # Couches du reseau

import numpy as np



# 1 : Tokenization
def rnn_tokenization(sentences):
    """  
    Process RNN Feed forward process
    """
    tokenizer = Tokenizer() # Create tokenizer
    tokenizer.fit_on_texts(sentences) # Apprentissage du vocabulaire a partir du corpus
    vocab_size = len(tokenizer.word_index) + 1 # Nombre total de mots (+1 pour l'indice 0)
    print(f"\n Taille du vocabulaire =>: {vocab_size}\n ")
    return tokenizer, vocab_size

# 2 Data preparation
def rnn_data_preparation(sentenses_indeces):
    """ Model process """
    X = [] # Listes des contextes (entrées)
    y = [] # Listes des mots cibles (sorties)
    # Parcours de chaque phrase du corpus
    max_len = max(len(seq) for seq in sentenses_indeces) # Longueur maximale des phrases du corpus
    X = pad_sequences(sentenses_indeces, maxlen=max_len, padding="pre") # Adjout de padding pour uniformiser la longueur
    
    y = X[:, -1] # Le dernier motde chaque sequence est la cible à predire
    X = X[:, :-1] # Les mots précedents constituent l'entree du RNN
    return X, y, max_len

# 3 Model preparation
def rnn_model(vocab_size, max_len):
    """Recurrent Neuronal Network
    Définition du réseau neurenal
    """
    # Definition du réseau
    model = Sequential([
        # Model sequentiel: les couches sont empilees
        Embedding(
            input_dim=vocab_size, # Taille  du  vocabulaire
            output_dim=8, # Dimension des embedins
            input_length=max_len-1, # Longueur des sequences d'entree 
        ),
        SimpleRNN(16), # Couche recurrente avec 16 unites (memoire interne)
        Dense(16, activation="relu"), # Couche cachée non lineaire, permet d'apprendre des relations plus complexes que de simples co-occurences.
        Dense(vocab_size, activation="softmax"), # Couche de sortie: probabilité pour chaque mot
    ])
    model.compile(
        optimizer="adam", #Methode d'optimisation des parametres qui ajuste les poids du RNN
        loss="sparse_categorical_crossentropy", # Fonction de perte (coût) pour classification muticlasse
        metrics= ["accuracy"], # Mesure de la performance du model
    )

    return model


# 4. Model training
def fit_rnn_model(model, x_data, y_data):
    """
    Entraînement du modèle
    
    :param model: Description
    :param x_data: Description
    :param y_data: Description

    """
    model.fit(
        x_data, 
        y_data, 
        epochs=200, 
        verbose=0) # Entrainement du model sur les données
    return model






if __name__ == "__main__":
    from w2v_pret import w2v_preprecess
    raw_sentenses = [
        "le chat que le chien a vu mange",
        "le chien que le chat poursuit aboie",
        "le chat que le voisin nourrit dort",
        "le chien que le voisin adopte mange",
        "le chat que le chien effraie se cache",
        "le chien que le chat observe se nourrit",
    ]
    sentenses = w2v_preprecess(rd=raw_sentenses, sw=False)
    print(f"Le sentense est =>: {sentenses}")
    print(f" RNN")
    # 1. Tokenizer
    tokenizer, vocab_size = rnn_tokenization(sentences=sentenses)
    setences_indices = tokenizer.texts_to_sequences(sentenses)

    #2. Data 
    x_data, y_data, max_len = rnn_data_preparation (sentenses_indeces=setences_indices)

    # 3. Model 
    model = rnn_model(vocab_size=vocab_size, max_len=max_len)

    # 4. Training
    model = fit_rnn_model(model=model, x_data=x_data, y_data=y_data)

    # Prediction
    test_sentence = ["le", "chat", "que", "le", "chien", "a", "vu"]
    # Conversion en indices
    test = tokenizer.texts_to_sequences([test_sentence])

    # Padding pour correspondre au format d'entree du model
    test = pad_sequences(test, maxlen=max_len-1, padding="pre")

    # Prediction
    prediction = model.predict(test)
    predicted_word = tokenizer.index_word[prediction.argmax()]
    print(f"Mot predit par RNN =>: {predicted_word}")
