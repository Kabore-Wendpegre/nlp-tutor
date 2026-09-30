"""
Application Gradio de NLP Tutor.

Cette application charge :

1. le tokenizer Qwen ;
2. le modèle de base quantifié ;
3. l'adaptateur LoRA entraîné ;
4. l'interface de conversation Gradio.

Le code d'entraînement n'est PAS exécuté ici.
L'application sert uniquement à l'inférence.
"""

from pathlib import Path

import gradio as gr
import torch

from nlp_tutor.model import (
    load_tokenizer,
    load_finetuned_model,
)

from nlp_tutor.generation import generate_tutor


# ============================================================
# CHEMINS DU PROJET
# ============================================================

# app.py se trouve dans :
#
# nlp_tutor/
# └── app/
#     └── app.py
#
# parent         -> app/
# parent.parent  -> nlp_tutor/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Dossier dans lequel le notebook sauvegarde
# l'adaptateur LoRA après le fine-tuning.
ADAPTER_DIR = PROJECT_ROOT / "artifacts" / "lora_adapter"


# ============================================================
# VÉRIFICATION DU GPU
# ============================================================

def check_environment():
    """
    Vérifie que l'environnement permet de charger le modèle QLoRA.

    Notre modèle est chargé en quantification 4 bits avec bitsandbytes.
    L'utilisation prévue dans ce projet est donc un environnement CUDA.
    """

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Aucun GPU CUDA détecté.\n"
            "NLP Tutor utilise ici un modèle QLoRA quantifié en 4 bits.\n"
            "Lance l'application sur un environnement GPU, "
            "par exemple Google Colab avec une Tesla T4."
        )

    print("CUDA disponible : True")
    print("GPU :", torch.cuda.get_device_name(0))


# ============================================================
# VÉRIFICATION DE L'ADAPTATEUR
# ============================================================

def check_adapter():
    """
    Vérifie que le fine-tuning a déjà été effectué.

    Le dossier artifacts/lora_adapter doit contenir les fichiers
    sauvegardés par PEFT après l'entraînement.
    """

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            "\nAdaptateur LoRA introuvable.\n\n"
            f"Dossier attendu : {ADAPTER_DIR}\n\n"
            "Il faut d'abord exécuter le fine-tuning dans "
            "notebooks/agent.ipynb puis sauvegarder l'adaptateur."
        )


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

def load_application_model():
    """
    Charge tout ce qui est nécessaire à l'inférence.

    Returns
    -------
    tuple
        tokenizer, model
    """

    print("=" * 70)
    print("Initialisation de NLP Tutor")
    print("=" * 70)

    # Vérification du GPU.
    check_environment()

    # Vérification de l'existence de l'adaptateur.
    check_adapter()

    print("\nChargement du tokenizer...")

    tokenizer = load_tokenizer()

    print("Tokenizer chargé.")

    print("\nChargement du modèle QLoRA...")

    # Cette fonction :
    #
    # 1. recharge le modèle Qwen de base ;
    # 2. applique la quantification 4 bits ;
    # 3. recharge les poids LoRA présents dans ADAPTER_DIR.
    model = load_finetuned_model(
        adapter_dir=ADAPTER_DIR,
    )

    model.eval()

    print("Modèle QLoRA chargé.")
    print("=" * 70)

    return tokenizer, model


# ============================================================
# CHARGEMENT UNIQUE AU DÉMARRAGE
# ============================================================

# Très important :
#
# nous chargeons le modèle UNE SEULE FOIS.
#
# Il ne faut surtout pas faire :
#
#     load_finetuned_model(...)
#
# à chaque question utilisateur.
#
# Sinon plusieurs Go seraient rechargés à chaque requête.
TOKENIZER, MODEL = load_application_model()


# ============================================================
# FONCTION UTILISÉE PAR GRADIO
# ============================================================

def answer_question(
    message,
    history,
):
    """
    Produit la réponse de NLP Tutor.

    Parameters
    ----------
    message : str
        Question actuelle de l'utilisateur.

    history
        Historique fourni automatiquement par Gradio.

    Returns
    -------
    str
        Réponse générée par le modèle.
    """

    # --------------------------------------------------------
    # VALIDATION DE L'ENTRÉE
    # --------------------------------------------------------

    if message is None:
        return "Veuillez saisir une question."

    message = message.strip()

    if not message:
        return "Veuillez saisir une question."


    # --------------------------------------------------------
    # GÉNÉRATION
    # --------------------------------------------------------

    # Nous utilisons un peu d'échantillonnage pour rendre
    # l'assistant naturel sans augmenter excessivement
    # la variabilité.
    response = generate_tutor(
        message,
        MODEL,
        TOKENIZER,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
        max_new_tokens=250,
    )

    return response


# ============================================================
# INTERFACE GRADIO
# ============================================================

demo = gr.ChatInterface(

    # Fonction exécutée à chaque message.
    fn=answer_question,

    # Nom affiché en haut de l'application.
    title="NLP Tutor",

    # Présentation rapide.
    description=(
        "Assistant pédagogique francophone spécialisé dans "
        "le NLP, les modèles de langage et les Transformers. "
        "Le modèle de base Qwen2.5-1.5B-Instruct a été adapté "
        "avec QLoRA."
    ),

    # Questions permettant de tester rapidement l'application.
    examples=[
        "Qu'est-ce qu'un token en NLP ?",
        "Explique-moi simplement la self-attention.",
        "Quelle est la différence entre LoRA et QLoRA ?",
        "À quoi sert la température pendant la génération ?",
        "Quelle différence y a-t-il entre les logits et les probabilités ?",
        "Pourquoi utilise-t-on un jeu de validation ?",
    ],
)


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    demo.launch(
        show_error=True,
        share=True,
        debug=True,
    )