from pprint import pprint
from numpy import dot
from numpy.linalg import norm
from gensim.models import Word2Vec

from w2v_pret import w2v_preprecess


def training():
    """ Training by Word2Vec """
    sentenses = w2v_preprecess()
    model = Word2Vec(
        sentences=sentenses,
        vector_size=50, # Vector' size of one word neigboring
        window=2, # Mot a prendre avant et après le mot cible, 2 signifies prendre 1 avant et 1 après le mot cible
        min_count=1, # Signifie de considérer un mot si il apparaît au moins 1 fois, mettre 2 signifierait de prendre les mots sss il apparaît 2 fois
        sg=0 # 0 for CBoW // 1 for Skip-gram
    )
    chat_sim = model.wv.most_similar('chat',topn=20)
    print(f"Chat simalarity =>: \n")
    for n in chat_sim:
        print(f"{n}")
    print(f"\n")
    return model

def consine_sim(u,v):
    """  
    Calcule similarity between two word via thiers vetors by consine method

    """
    return dot(u,v)/(norm(u)*norm(v))



if __name__ == "__main__":
    model = training()
    print(f"Sim(chat, chien) = {round(consine_sim(model.wv['chat'], model.wv['chien']), 3 )}")
    print(f"Sim(voiture, automobile) = {round(consine_sim(model.wv['voiture'], model.wv['automobile']), 3 )}")
    print(f"Sim(chat, automobile) = {round(consine_sim(model.wv['chat'], model.wv['automobile']), 3 )}")