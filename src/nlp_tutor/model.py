"""
Chargement du tokenizer et des différents modèles utilisés par NLP Tutor.

Ce module centralise tout ce qui concerne le chargement de Qwen :

1. tokenizer ;
2. modèle de base FP16 pour la baseline ;
3. modèle quantifié 4 bits pour QLoRA ;
4. modèle fine-tuné rechargé avec son adaptateur LoRA.

Aucune génération et aucun entraînement ne sont réalisés ici.
"""

import torch

from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .config import MODEL_ID


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer(model_id: str = MODEL_ID):
    """
    Charge et configure le tokenizer associé au modèle.

    Parameters
    ----------
    model_id : str
        Identifiant Hugging Face du modèle.

    Returns
    -------
    tokenizer
        Tokenizer prêt à être utilisé.
    """

    # Télécharge / charge le tokenizer correspondant au modèle.
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Certains modèles de langage n'ont pas de token de padding
    # explicitement défini.
    #
    # Dans ce cas, on utilise EOS (End Of Sequence) comme padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Le padding sera ajouté à droite des séquences.
    #
    # Exemple :
    #
    # [token1, token2, token3, PAD, PAD]
    #
    # C'est adapté à notre fine-tuning causal.
    tokenizer.padding_side = "right"

    return tokenizer


# ============================================================
# MODÈLE DE BASE
# ============================================================

def load_base_model(model_id: str = MODEL_ID):
    """
    Charge Qwen en float16 sur le GPU.

    Cette version est utilisée pour :

    - la baseline avant fine-tuning ;
    - l'étude des logits ;
    - les expériences greedy / top-k / top-p / température.

    Aucun adaptateur LoRA n'est présent ici.
    """

    # Vérification explicite du GPU.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Aucun GPU CUDA détecté. "
            "Le projet a été conçu pour être exécuté sur GPU."
        )

    # Chargement du modèle.
    #
    # dtype=torch.float16 :
    # réduit la mémoire par rapport au float32.
    #
    # device_map={"": 0} :
    # place tout le modèle sur le GPU CUDA numéro 0.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map={"": 0},
    )

    # Mode évaluation :
    # désactive notamment les comportements spécifiques
    # à l'entraînement comme le dropout.
    model.eval()

    return model


# ============================================================
# CONFIGURATION 4 BITS
# ============================================================

def build_4bit_config():
    """
    Construit la configuration bitsandbytes utilisée par QLoRA.

    QLoRA repose ici sur une quantification du modèle de base
    en 4 bits afin de réduire fortement l'utilisation de VRAM.
    """

    quantization_config = BitsAndBytesConfig(

        # Les poids du modèle de base sont chargés en 4 bits.
        load_in_4bit=True,

        # NF4 = NormalFloat 4.
        #
        # Format particulièrement adapté aux poids de réseaux
        # neuronaux utilisés avec QLoRA.
        bnb_4bit_quant_type="nf4",

        # Quantifie aussi certaines constantes de quantification.
        #
        # Cela permet de gagner encore un peu de mémoire.
        bnb_4bit_use_double_quant=True,

        # Même si les poids sont stockés en 4 bits,
        # les calculs intermédiaires sont effectués en float16.
        #
        # C'est particulièrement adapté à la Tesla T4.
        bnb_4bit_compute_dtype=torch.float16,
    )

    return quantization_config


# ============================================================
# MODÈLE QUANTIFIÉ POUR L'ENTRAÎNEMENT QLoRA
# ============================================================

def load_qlora_base_model(model_id: str = MODEL_ID):
    """
    Charge le modèle de base quantifié en 4 bits
    et le prépare pour l'entraînement QLoRA.

    Attention :
    les adaptateurs LoRA ne sont pas encore ajoutés ici.

    Ils seront ajoutés plus tard dans training.py via SFTTrainer.
    """

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Aucun GPU CUDA détecté. "
            "QLoRA nécessite ici un runtime GPU."
        )

    # Chargement du même Qwen,
    # mais cette fois avec quantification 4 bits.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=build_4bit_config(),
        dtype=torch.float16,
        device_map={"": 0},
    )

    # Pendant l'entraînement, le cache des activations
    # est incompatible / inutile avec le gradient checkpointing.
    model.config.use_cache = False

    # Cette fonction PEFT prépare certaines couches du modèle
    # quantifié afin de permettre un entraînement stable en k-bit.
    model = prepare_model_for_kbit_training(model)

    return model


# ============================================================
# MODÈLE FINE-TUNÉ
# ============================================================

def load_finetuned_model(
    adapter_dir: str,
    model_id: str = MODEL_ID,
):
    """
    Recharge un modèle déjà entraîné avec QLoRA.

    Le modèle final est constitué de deux éléments :

        Qwen original quantifié
                +
        adaptateur LoRA entraîné

    Parameters
    ----------
    adapter_dir : str
        Chemin vers le dossier contenant l'adaptateur LoRA.

    model_id : str
        Modèle de base Hugging Face.

    Returns
    -------
    model
        Modèle QLoRA prêt pour l'inférence.
    """

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Aucun GPU CUDA détecté."
        )

    # --------------------------------------------------------
    # 1. Recharger le modèle original en 4 bits
    # --------------------------------------------------------

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=build_4bit_config(),
        dtype=torch.float16,
        device_map={"": 0},
    )

    # --------------------------------------------------------
    # 2. Ajouter l'adaptateur LoRA entraîné
    # --------------------------------------------------------

    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
    )

    # Pour l'inférence, le cache peut être réactivé.
    model.config.use_cache = True

    # Passage en mode évaluation.
    model.eval()

    return model