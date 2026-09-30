"""
Fonctions de génération de texte pour NLP Tutor.

Ce module contient toute la logique permettant de passer :

    question utilisateur
            ↓
    messages conversationnels
            ↓
    tokenisation
            ↓
    génération par le modèle
            ↓
    décodage
            ↓
    réponse finale

Il ne charge PAS le modèle.
Il ne construit PAS le dataset.
Il n'entraîne PAS le modèle.

Il reçoit simplement un modèle et un tokenizer déjà chargés.
"""

import torch

from .config import (
    MAX_NEW_TOKENS,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
)


# ============================================================
# UTILITAIRE : TROUVER LE DEVICE DU MODÈLE
# ============================================================

def get_model_device(model):
    """
    Retourne le device sur lequel se trouve le modèle.

    Exemple :
        cuda:0
        cpu

    Pourquoi cette fonction ?

    Les tenseurs d'entrée et le modèle doivent se trouver sur le
    même device.

    Si le modèle est sur le GPU mais que les données restent sur
    le CPU, PyTorch déclenchera une erreur.
    """

    return next(model.parameters()).device


# ============================================================
# GÉNÉRATION PRINCIPALE DE NLP TUTOR
# ============================================================

def generate_tutor(
    question: str,
    model,
    tokenizer,
    do_sample: bool = False,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    """
    Génère la réponse pédagogique de NLP Tutor.

    Parameters
    ----------
    question : str
        Question posée par l'utilisateur.

    model
        Modèle de langage déjà chargé.

        Cela peut être :
        - le modèle de base ;
        - le modèle QLoRA fine-tuné.

    tokenizer
        Tokenizer correspondant au modèle.

    do_sample : bool
        False :
            génération déterministe.

        True :
            génération par échantillonnage.

    temperature : float
        Contrôle la dispersion des probabilités pendant
        l'échantillonnage.

    top_p : float
        Utilise le nucleus sampling.

    max_new_tokens : int
        Nombre maximal de nouveaux tokens générés.

    Returns
    -------
    str
        Réponse générée par le modèle.
    """

    # --------------------------------------------------------
    # 1. CONSTRUIRE LA CONVERSATION
    # --------------------------------------------------------

    # Les modèles "Instruct" utilisent généralement un format
    # conversationnel composé de rôles.
    #
    # Ici :
    #
    # system -> définit le comportement du modèle
    # user   -> contient la question
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]


    # --------------------------------------------------------
    # 2. APPLIQUER LE CHAT TEMPLATE
    # --------------------------------------------------------

    # On ne doit PAS construire manuellement quelque chose comme :
    #
    # "System: ... User: ... Assistant:"
    #
    # Chaque famille de modèles possède son propre format.
    #
    # apply_chat_template() utilise celui qui est fourni avec
    # le tokenizer Qwen.
    model_inputs = tokenizer.apply_chat_template(
        messages,

        # Ajoute le marqueur indiquant que le prochain message
        # doit être produit par l'assistant.
        add_generation_prompt=True,

        # On veut directement obtenir les tokens.
        tokenize=True,

        # Retour sous forme de dictionnaire :
        #
        # {
        #     "input_ids": ...,
        #     "attention_mask": ...
        # }
        return_dict=True,

        # Conversion en tenseurs PyTorch.
        return_tensors="pt",
    )


    # --------------------------------------------------------
    # 3. ENVOYER LES ENTRÉES SUR LE BON DEVICE
    # --------------------------------------------------------

    device = get_model_device(model)

    model_inputs = model_inputs.to(device)


    # --------------------------------------------------------
    # 4. MÉMORISER LA LONGUEUR DU PROMPT
    # --------------------------------------------------------

    # Exemple simplifié :
    #
    # entrée :
    # [SYSTEM, USER, question]
    #
    # après generate() :
    #
    # [SYSTEM, USER, question, REPONSE]
    #
    # Pour ne récupérer que REPONSE, on mémorise donc la longueur
    # du prompt avant la génération.
    input_length = model_inputs["input_ids"].shape[-1]


    # --------------------------------------------------------
    # 5. PARAMÈTRES COMMUNS DE GÉNÉRATION
    # --------------------------------------------------------

    generation_kwargs = {

        # Nombre maximal de tokens produits après le prompt.
        "max_new_tokens": max_new_tokens,

        # False = greedy
        # True  = sampling
        "do_sample": do_sample,

        # Légère pénalité destinée à limiter les répétitions.
        "repetition_penalty": 1.05,

        # Token utilisé si un padding est nécessaire.
        "pad_token_id": tokenizer.eos_token_id,
    }


    # --------------------------------------------------------
    # 6. PARAMÈTRES UTILISÉS UNIQUEMENT AVEC LE SAMPLING
    # --------------------------------------------------------

    # temperature et top_p n'ont réellement de sens ici que lorsque
    # l'on échantillonne.
    if do_sample:

        generation_kwargs["temperature"] = temperature

        generation_kwargs["top_p"] = top_p


    # --------------------------------------------------------
    # 7. GÉNÉRER
    # --------------------------------------------------------

    # inference_mode indique à PyTorch :
    #
    # "nous faisons uniquement de l'inférence,
    #  inutile de calculer les gradients."
    #
    # Cela réduit l'utilisation mémoire.
    with torch.inference_mode():

        output_ids = model.generate(
            **model_inputs,
            **generation_kwargs,
        )


    # --------------------------------------------------------
    # 8. RETIRER LE PROMPT DE LA SORTIE
    # --------------------------------------------------------

    # output_ids contient :
    #
    # prompt + réponse
    #
    # Nous ne voulons afficher que la partie générée.
    generated_ids = output_ids[
        0,
        input_length:
    ]


    # --------------------------------------------------------
    # 9. CONVERTIR LES TOKENS EN TEXTE
    # --------------------------------------------------------

    answer = tokenizer.decode(
        generated_ids,

        # On ne souhaite pas afficher :
        # <|endoftext|>, etc.
        skip_special_tokens=True,
    )


    # Suppression des espaces inutiles au début et à la fin.
    return answer.strip()


# ============================================================
# COMPARAISON DES STRATÉGIES DE DÉCODAGE
# ============================================================

def generate_with_strategy(
    question: str,
    model,
    tokenizer,
    *,
    do_sample: bool = False,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    max_new_tokens: int = 120,
):
    """
    Génère une réponse avec une stratégie de décodage donnée.

    Cette fonction est principalement pédagogique.

    Elle permet de comparer :

        greedy
        température
        top-k
        top-p

    dans le notebook.
    """

    # Pour cette expérience, pas besoin du SYSTEM_PROMPT.
    #
    # On veut surtout observer l'effet de la stratégie
    # de génération.
    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]


    # Transformation de la conversation en tokens Qwen.
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


    # Envoi vers le même device que le modèle.
    device = get_model_device(model)

    model_inputs = model_inputs.to(device)


    # Longueur du prompt avant génération.
    input_length = model_inputs["input_ids"].shape[-1]


    # Paramètres communs.
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }


    # --------------------------------------------------------
    # PARAMÈTRES DE SAMPLING
    # --------------------------------------------------------

    if do_sample:

        # Température :
        #
        # faible -> distribution plus concentrée
        # haute  -> distribution plus plate
        if temperature is not None:
            generation_kwargs["temperature"] = temperature


        # Top-k :
        #
        # ne conserve que les k tokens les plus probables.
        if top_k is not None:
            generation_kwargs["top_k"] = top_k


        # Top-p :
        #
        # conserve le plus petit ensemble de tokens dont
        # la probabilité cumulée atteint p.
        if top_p is not None:
            generation_kwargs["top_p"] = top_p


    # Génération sans gradient.
    with torch.inference_mode():

        output_ids = model.generate(
            **model_inputs,
            **generation_kwargs,
        )


    # On garde uniquement les nouveaux tokens.
    generated_ids = output_ids[
        0,
        input_length:
    ]


    # Conversion tokens -> texte.
    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()