"""
Configuration centrale du projet NLP Tutor.

Ce module regroupe les constantes partagées par plusieurs parties
du projet : modèle, génération, entraînement, etc.

"""


# 
# MODÈLE
# 

# Identifiant Hugging Face du modèle utilisé dans tout le projet.
#
# "Instruct" signifie que cette version de Qwen a déjà été adaptée
# pour suivre des instructions et dialoguer avec un utilisateur.
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


# 
# REPRODUCTIBILITÉ
# 

# Graine aléatoire commune.
#

SEED = 42


# 
# PROMPT SYSTÈME
# 

# Ce prompt définit le rôle attendu de notre assistant.
#
# Il sera réutilisé :
# - pour la baseline ;
# - pour construire le dataset conversationnel ;
# - pour l'inférence finale.
SYSTEM_PROMPT = (
    "Tu es NLP Tutor, un assistant pédagogique francophone spécialisé en NLP, "
    "modèles de langage et Transformers.\n"
    "Règles :\n"
    "1. Réponds en français.\n"
    "2. Commence par une définition claire.\n"
    "3. Explique progressivement avec des mots simples.\n"
    "4. Donne au moins un exemple concret quand c'est pertinent.\n"
    "5. Termine par une phrase 'À retenir : ...'.\n"
    "6. N'invente pas une information si tu n'es pas sûr."
)


# 
# PARAMÈTRES DE GÉNÉRATION
# 

# Nombre maximal de nouveaux tokens que le modèle peut générer
# pour une réponse.
MAX_NEW_TOKENS = 220


# Température utilisée lorsque l'on active l'échantillonnage.
#
# Une valeur relativement basse donne des réponses plus stables
# et convient bien à un assistant pédagogique.
TEMPERATURE = 0.4


# Paramètre du nucleus sampling (top-p).
#
# Le modèle conserve un ensemble de tokens dont la probabilité
# cumulée atteint 90 %.
TOP_P = 0.9


# 
# PARAMÈTRES LoRA
# 

# Rang des petites matrices entraînables LoRA.
#
# Un rang plus grand donne davantage de capacité d'adaptation,
# mais augmente également le nombre de paramètres entraînables.
LORA_R = 16


# Facteur d'échelle appliqué aux mises à jour LoRA.
LORA_ALPHA = 32


# Dropout appliqué dans les adaptateurs LoRA afin d'introduire
# une légère régularisation.
LORA_DROPOUT = 0.05


# 
# PARAMÈTRES D'ENTRAÎNEMENT
# 

# Nombre de passages complets sur le dataset d'entraînement.
TRAIN_EPOCHS = 2


# Nombre d'exemples traités simultanément par le GPU.
TRAIN_BATCH_SIZE = 2


# Batch utilisé pendant la validation.
EVAL_BATCH_SIZE = 2


# Nombre de mini-batchs dont on accumule les gradients avant
# d'effectuer une mise à jour des paramètres.
#
# Avec :
# batch_size = 2
# accumulation = 8
#
# le batch effectif vaut :
# 2 × 8 = 16 exemples.
GRADIENT_ACCUMULATION_STEPS = 8


# Taux d'apprentissage appliqué aux paramètres LoRA.
LEARNING_RATE = 2e-4


# Longueur maximale d'un exemple après tokenisation.
MAX_LENGTH = 512