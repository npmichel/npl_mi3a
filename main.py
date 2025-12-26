"""
Point d'entrée du projet NLP

Pour exécuter toutes les démonstrations, utiliser plutôt:
    uv run python pre_processing.py

Ce fichier main.py peut être utilisé pour tester des fonctions individuelles.
"""
import pre_processing


def main():
    """Exemple d'utilisation des fonctions de prétraitement"""

    print("Hello from npl-mi3a!")
    print("\n📌 Pour exécuter toutes les démonstrations, utilisez:")
    print("   uv run python pre_processing.py\n")

    # Exemple simple d'utilisation
    text = """
    La Coupe d'Afrique des Nations 2025 (CAN 2025) se tiendra au Maroc ! C'est
    une compétition très attendue par les fans de football. L'événement débutera officiellement en décembre 2025, avec des matchs dans plusieurs villes :
    Casablanca, Rabat, Marrakech, Fès et Tanger. Les organisateurs annoncent déjà plus de 100.000 billets vendus en ligne.
    Le site officiel (https ://www.cafonline.com)
    propose toutes les informations nécessaires. Contact presse : info@can2025.ma
    Cependant, certains supporters s'inquiètent du prix des tickets, jugés parfois «
    trop élevés ». Le Maroc, qui avait déjà accueilli de grands événements sportifs
    internationaux, espère renforcer son image et montrer sa capacité d'organisation.
    Les Lions de l'Atlas, quant à eux, rêvent de soulever le trophée devant leur public !
    """

    print("Exemple rapide de quelques fonctions:\n")

    # Normalisation
    normalized = pre_processing.normalizing(text)

    # Tokenisation
    tokens = pre_processing.spacy_tokenization(normalized)

    # Lemmatisation
    lemmas = pre_processing.spacy_lemmatization(normalized)

    # NER
    entities = pre_processing.named_entity_recognition(text)


if __name__ == "__main__":
    main()
