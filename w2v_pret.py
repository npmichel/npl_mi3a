import  re
from  nltk.tokenize  import  word_tokenize
from  nltk.corpus  import  stopwords
import  nltk

# Télécharger uniquement si pas déjà présent
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")


def w2v_preprecess(rd: list[str] = None, sw = True):
    """
    Docstring pour w2v_preprecess
    """
    raw_sentences = [
        "Le chat  mange  une  souris",
        "Le chat  boit du lait",
        "Le chien  mange  une  viande",
        "Le chien  aboie  souvent",
        "L'âne boit  de l'eau",
        "Le chat et le chien  sont  des  animaux",
        "La  voiture  est un moyen de  transport",
        "L'automobile  est un  moyen de  transport"]
    if rd:
        raw_sentences = rd
    sentences = []
    if sw:
        stop_words_list = set(stopwords.words("french"))
        for s in  raw_sentences:
            s = s.lower()
            s = re.sub(r'[^a-z\s]', ' ', s)
            tokens = word_tokenize(s, language="french")
            tokens = [w for w in  tokens  if w not in stop_words_list]
            sentences.append(tokens)
    else:
        for s in  raw_sentences:
            s = s.lower()
            s = re.sub(r'[^a-z\s]', ' ', s)
            tokens = word_tokenize(s, language="french")
            tokens = [w for w in  tokens]
            sentences.append(tokens)
    print(sentences)

    return sentences


if __name__ == "__main__":
    w2v_preprecess()