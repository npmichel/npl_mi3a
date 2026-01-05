# Importation du model pre entrainer
from transformers import   GPT2LMHeadModel , GPT2Tokenizer

# Chargement du tokenizer associe a GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Chargement du modele GTP-2
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")


def reformuler_text(texte:str, max_length=80):
    """ Fonction permettant de reformuler du texte """
    prompt = (
        f"Explain in simple terms the following concept: {texte}"
    )
    # Encodage du texte en tokens
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generation du texte
    outputs = gpt2_model.generate(
        inputs,
        max_length=max_length, # # Longueur  maximale  du texte  genEre (prompt  inclus)
        do_sample=True, # Active  un  decodage  probabiliste (echantillonnage)
        top_k=50, # Limite  le choix du  prochain  token  aux50 plus  probables
        top_p=0.95, # selectionne  le plus  petit  ensemble  detokens  dont la  somme  des  probabilites  atteint  95 %
    )

    # Decodage des tokens vers du texte lisible
    return tokenizer.decode(outputs[0],  skip_special_tokens=True)



resultat = reformuler_text(texte="What is an electron?")
print(resultat)
