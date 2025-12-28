from nltk.util import ngrams
from collections import Counter


def ngram_process( sentences: list[str], ng=2):
    """
    Training with n grammes
    """
    ngrams_list = []
    for sent in sentences:
        ngrams_list.extend(list(ngrams(sent, ng)))
    
    # Count frequency
    ng_frq = Counter(ngrams_list)
    # Show most ng frq
    # print(f"\n Most ng =>: \n {ng_frq.most_common(10)}")

    return ng_frq


def prob_ngram(w2:str, w1: str, ngf: Counter, cn: Counter):
    """ ngrams probability """
    return ngf[(w1, w2)]/cn[w1]




if __name__ == "__main__":
    from w2v_pret import w2v_preprecess
    sentenses = w2v_preprecess()

    print(f"\n Bigram process (n = 2) \n ")
    bigrams = ngram_process(sentences=sentenses)

    unigrams = Counter([w for sent in sentenses for w in sent])
    print(f"\n P(mange | chat ) = {round(prob_ngram(w2= "mange", w1="chat", ngf=bigrams, cn=unigrams), 3)} \n ")
    print(f"\n P(transport | moyen ) = {round(prob_ngram(w2= "transport", w1="moyen", ngf=bigrams, cn=unigrams), 3)} \n ")


    print(f"\n Trigram process (n = 3) \n ")
    trigrams = ngram_process(sentences=sentenses, ng=3)


