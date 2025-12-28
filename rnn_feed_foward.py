from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer # pyright: ignore[reportMissingImports] # Tokenization tools
from tensorflow.keras.models import Sequential # type: ignore # Model Sequentiel
from tensorflow.keras.layers import Embedding , Flatten , Dense # type: ignore # Couches du reseau

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
def data_preparation(sentenses: list[str], tokenizer: Tokenizer):
    """ Model process """
    X = [] # Listes des contextes (entrées)
    y = [] # Listes des mots cibles (sorties)
    # Parcours de chaque phrase du corpus
    for sent in sentenses:
        encoded = tokenizer.texts_to_sequences([sent])[0] # Conversion des mots en indices
        X.append(encoded[:2]) # Ajout des deux premiers mots comme contexte
        y.append(encoded[2]) # Ajout du troisième mot comme cile a prédire
    X = np.array(X) # Converstion des contextes en tableau Numpy
    y= np.array(y) # Conversion des mots cibles en tableau Numpy
    return X, y


# 3 Model preparation
def nn_feed_foward_model(vocab_size):
    """ Neuronal feed forward 
    Définition du réseau feed-forward
    """
    # Definition du réseau
    model = Sequential([
        Embedding(
            input_dim=vocab_size, # Taille  du  vocabulaire
            output_dim=8, # Dimension des embedins
            input_length=2, # Taille fixe du contexte 
        ),
        Flatten(), # Passage de 2D a 1D
        Dense(16, activation="relu"), # Couche cachée non lineaire, permet d'apprendre des relations plus complexes que de simples co-occurences.
        Dense(vocab_size, activation="softmax"), # Couche de sortie (Probabilistes)
    ])
    model.compile(
        optimizer="adam", #Methode d'optimisation des parametres
        loss="sparse_categorical_crossentropy", # Fonction de perte (coût) pour classification muticlasse
        metrics= ["accuracy"], # Mesure de la performance du model
    )

    return model


# 4. Model training
def fit_model(model, x_data, y_data):
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
        "le chat mange",
        "le chien mange",
        "le chat se nourrit",
        "le chien se nourrit",
        "le chat dort",
        "le chien aboie",
    ]
    sentenses = w2v_preprecess(rd=raw_sentenses, sw=False)
    print(f"Le sentense est =>: {sentenses}")
    print(f"Test feed-forward")
    # 1. Tokenizer
    tokenizer, vocab_size = rnn_tokenization(sentences=sentenses)

    #2. Data 
    x_data, y_data = data_preparation(sentenses=sentenses, tokenizer=tokenizer)

    # 3. Model 
    model = nn_feed_foward_model(vocab_size=vocab_size)

    # 4. Training
    model = fit_model(model=model, x_data=x_data, y_data=y_data)

    # Question 6
    test = tokenizer.texts_to_sequences([["le", "chat"]]) # Convertion du contexte en indices
    prediction = model.predict(np.array(test)) # Prediction des probabilités
    predicted_word = tokenizer.index_word[prediction.argmax()] # Mot avec probabilité maximale
    print(f"Mot prédit =>: {predicted_word}")

    # Question 7
    test_g = tokenizer.texts_to_sequences([["le", "chien"]])
    npred = model.predict(np.array(test_g))

    npred_word = tokenizer.index_word[npred.argmax()] # Mot avec probabilité maximale
    print(f"Nouveau mot prédit =>: {npred_word}")
    top_indices = npred[0].argsort()[-3:][::-1]
    top_words = [tokenizer.index_word[i] for i in top_indices]
    print(f"Mots predits =>: {top_words}") 

    print(f"Comparaison avec Ngram")
    from ngramme import ngram_process, prob_ngram

    print(f"\n Bigram process (n = 2) \n ")
    bigrams = ngram_process(sentences=sentenses)

    unigrams = Counter([w for sent in sentenses for w in sent])
    print(f"\n P(mange | chien ) = {round(prob_ngram(w2= "mange", w1="chien", ngf=bigrams, cn=unigrams), 3)} \n ")


