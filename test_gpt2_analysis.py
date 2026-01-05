"""
Script de test et d'analyse approfondie de GPT-2 pour le TP7
Teste différents aspects: cohérence, qualité, hallucinations, contexte éducatif
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

# Initialisation
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

print("=" * 80)
print("ANALYSE DE GPT-2 DANS UN CONTEXTE ÉDUCATIF - TP7")
print("=" * 80)

def generer_texte(prompt, max_length=80, temperature=1.0, top_k=50, top_p=0.95):
    """Génère du texte avec GPT-2"""
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Test 1: Cohérence et hallucinations
print("\n\n### TEST 1: COHÉRENCE ET HALLUCINATIONS ###\n")
prompts_educatifs = [
    "Explain the concept of photosynthesis:",
    "What is the capital of France?",
    "Define machine learning:",
    "The speed of light is"
]

for prompt in prompts_educatifs:
    print(f"\nPrompt: {prompt}")
    result = generer_texte(prompt, max_length=100)
    print(f"Réponse: {result}")
    print("-" * 80)


# Test 2: Influence de la température
print("\n\n### TEST 2: INFLUENCE DE LA TEMPÉRATURE ###\n")
prompt_test = "The solar system consists of"

for temp in [0.3, 0.7, 1.0, 1.5]:
    print(f"\nTempérature: {temp}")
    result = generer_texte(prompt_test, max_length=80, temperature=temp)
    print(f"Réponse: {result}")
    print("-" * 80)


# Test 3: Longueur de génération et cohérence
print("\n\n### TEST 3: COHÉRENCE AVEC DIFFÉRENTES LONGUEURS ###\n")
prompt_long = "Artificial intelligence is"

for length in [50, 100, 200]:
    print(f"\nLongueur max: {length}")
    result = generer_texte(prompt_long, max_length=length, temperature=0.7)
    print(f"Réponse: {result}")
    print(f"Longueur réelle: {len(result.split())}")
    print("-" * 80)


# Test 4: Questions complexes (niveau Master)
print("\n\n### TEST 4: QUESTIONS DE NIVEAU MASTER ###\n")
questions_complexes = [
    "Explain the backpropagation algorithm in neural networks:",
    "What is the difference between LSTM and GRU?",
    "Describe the attention mechanism in transformers:"
]

for question in questions_complexes:
    print(f"\nQuestion: {question}")
    result = generer_texte(question, max_length=150, temperature=0.7)
    print(f"Réponse: {result}")
    print("-" * 80)


# Test 5: Contexte français
print("\n\n### TEST 5: CAPACITÉS EN FRANÇAIS ###\n")
prompts_francais = [
    "Qu'est-ce qu'un réseau de neurones?",
    "Expliquez l'intelligence artificielle:"
]

for prompt in prompts_francais:
    print(f"\nPrompt (français): {prompt}")
    result = generer_texte(prompt, max_length=100, temperature=0.7)
    print(f"Réponse: {result}")
    print("-" * 80)


# Test 6: Répétabilité
print("\n\n### TEST 6: RÉPÉTABILITÉ (3 GÉNÉRATIONS IDENTIQUES) ###\n")
prompt_rep = "Machine learning is defined as"
print(f"Prompt: {prompt_rep}\n")

for i in range(3):
    print(f"Génération {i+1}:")
    result = generer_texte(prompt_rep, max_length=80, temperature=0.7)
    print(f"{result}")
    print("-" * 40)


print("\n\n" + "=" * 80)
print("FIN DE L'ANALYSE")
print("=" * 80)
