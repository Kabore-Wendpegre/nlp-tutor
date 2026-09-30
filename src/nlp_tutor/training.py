"""
Configuration et construction du fine-tuning QLoRA de NLP Tutor.

Ce module contient :

1. la configuration LoRA ;
2. les hyperparamètres d'entraînement ;
3. la construction du SFTTrainer.

Le chargement du modèle quantifié est réalisé dans model.py.
La construction du dataset est réalisée dans dataset.py.
L'évaluation est réalisée dans evaluation.py.
"""

from peft import LoraConfig

from trl import (
    SFTConfig,
    SFTTrainer,
)

from .config import (
    EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_LENGTH,
    SEED,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
)


# ============================================================
# CONFIGURATION LoRA
# ============================================================

def build_lora_config():
    """
    Construit la configuration des adaptateurs LoRA.

    LoRA ne modifie pas directement tous les poids du modèle.

    À la place :

        modèle original
              ↓
        poids principaux gelés
              +
        petites matrices LoRA entraînables

    Cela permet d'adapter un grand modèle avec beaucoup moins
    de paramètres à entraîner.

    Returns
    -------
    LoraConfig
        Configuration PEFT utilisée par le trainer.
    """

    return LoraConfig(

        # ----------------------------------------------------
        # RANG DES MATRICES LoRA
        # ----------------------------------------------------

        # r définit la dimension interne des petites matrices
        # ajoutées par LoRA.
        #
        # Plus r est élevé :
        # - plus LoRA peut représenter de modifications ;
        # - mais plus le nombre de paramètres augmente.
        r=LORA_R,


        # ----------------------------------------------------
        # FACTEUR D'ÉCHELLE
        # ----------------------------------------------------

        # lora_alpha contrôle l'amplitude des mises à jour LoRA.
        #
        # Une façon simplifiée de voir le facteur d'échelle est :
        #
        # alpha / r
        #
        # ici :
        #
        # 32 / 16 = 2
        lora_alpha=LORA_ALPHA,


        # ----------------------------------------------------
        # DROPOUT
        # ----------------------------------------------------

        # Une petite régularisation appliquée aux branches LoRA.
        lora_dropout=LORA_DROPOUT,


        # ----------------------------------------------------
        # BIAIS
        # ----------------------------------------------------

        # On n'entraîne pas les biais du modèle original.
        bias="none",


        # ----------------------------------------------------
        # TYPE DE MODÈLE
        # ----------------------------------------------------

        # Qwen est ici utilisé comme modèle causal :
        #
        # tokens précédents
        #       ↓
        # prédiction du token suivant
        task_type="CAUSAL_LM",


        # ----------------------------------------------------
        # COUCHES À ADAPTER
        # ----------------------------------------------------

        # Les projections de l'attention :
        #
        # q_proj -> Query
        # k_proj -> Key
        # v_proj -> Value
        # o_proj -> sortie de l'attention
        #
        # et les projections du MLP :
        #
        # gate_proj
        # up_proj
        # down_proj
        #
        # reçoivent des adaptateurs LoRA.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


# ============================================================
# CONFIGURATION DE L'ENTRAÎNEMENT
# ============================================================

def build_training_args(
    output_dir: str,
):
    """
    Construit les hyperparamètres du supervised fine-tuning.

    Les valeurs utilisées ici correspondent à l'expérience
    réalisée avec une Tesla T4.

    Parameters
    ----------
    output_dir : str
        Dossier dans lequel enregistrer les checkpoints.

    Returns
    -------
    SFTConfig
        Configuration TRL du fine-tuning.
    """

    return SFTConfig(

        # ----------------------------------------------------
        # SORTIES
        # ----------------------------------------------------

        output_dir=output_dir,


        # ----------------------------------------------------
        # NOMBRE D'EPOCHS
        # ----------------------------------------------------

        # Une epoch correspond à un passage complet sur
        # les 350 exemples d'entraînement.
        num_train_epochs=TRAIN_EPOCHS,


        # ----------------------------------------------------
        # BATCH SIZE
        # ----------------------------------------------------

        # Nombre d'exemples réellement présents simultanément
        # dans un mini-batch GPU.
        per_device_train_batch_size=TRAIN_BATCH_SIZE,

        per_device_eval_batch_size=EVAL_BATCH_SIZE,


        # ----------------------------------------------------
        # ACCUMULATION DES GRADIENTS
        # ----------------------------------------------------

        # Notre T4 ne peut pas forcément traiter un gros batch
        # simultanément.
        #
        # On traite donc plusieurs petits batchs avant de faire
        # une mise à jour des paramètres.
        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),


        # ----------------------------------------------------
        # LEARNING RATE
        # ----------------------------------------------------

        # Vitesse avec laquelle les paramètres LoRA sont modifiés.
        learning_rate=LEARNING_RATE,


        # ----------------------------------------------------
        # WARMUP
        # ----------------------------------------------------

        # Pendant les premières étapes, le learning rate augmente
        # progressivement avant d'atteindre sa valeur normale.
        warmup_steps=10,


        # ----------------------------------------------------
        # LOGS
        # ----------------------------------------------------

        # Affiche les métriques toutes les 10 étapes.
        logging_steps=10,


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        # Une évaluation sur le split validation est effectuée
        # à la fin de chaque epoch.
        eval_strategy="epoch",


        # ----------------------------------------------------
        # CHECKPOINTS
        # ----------------------------------------------------

        # Sauvegarde également à chaque epoch.
        save_strategy="epoch",

        # Évite de conserver trop de checkpoints.
        save_total_limit=2,


        # ----------------------------------------------------
        # PRÉCISION NUMÉRIQUE DU TRAINER
        # ----------------------------------------------------

        # Lors de notre première expérience, l'activation de fp16
        # dans SFTTrainer provoquait un conflit de mixed precision
        # sur le runtime T4.
        #
        # On les désactive donc ici.
        #
        # IMPORTANT :
        # cela ne supprime pas la quantification 4 bits.
        # Le modèle chargé dans model.py reste bien quantifié.
        fp16=False,
        bf16=False,


        # ----------------------------------------------------
        # GRADIENT CHECKPOINTING
        # ----------------------------------------------------

        # Au lieu de conserver toutes les activations en mémoire,
        # certaines seront recalculées pendant le backward pass.
        #
        # Avantage :
        # moins de VRAM.
        #
        # Inconvénient :
        # davantage de calcul.
        gradient_checkpointing=True,


        # ----------------------------------------------------
        # LONGUEUR MAXIMALE
        # ----------------------------------------------------

        # Les exemples plus longs sont tronqués à cette taille.
        max_length=MAX_LENGTH,


        # ----------------------------------------------------
        # LOSS SUR LA COMPLÉTION UNIQUEMENT
        # ----------------------------------------------------

        # Notre exemple contient :
        #
        # SYSTEM
        # USER
        # ASSISTANT
        #
        # Nous souhaitons que le modèle apprenne surtout à produire
        # la réponse assistant.
        #
        # La loss n'est donc calculée que sur la completion.
        completion_only_loss=True,


        # ----------------------------------------------------
        # OPTIMISEUR
        # ----------------------------------------------------

        optim="adamw_torch",


        # ----------------------------------------------------
        # TRACKING EXTERNE
        # ----------------------------------------------------

        # On désactive WandB et autres outils automatiques.
        report_to="none",


        # ----------------------------------------------------
        # REPRODUCTIBILITÉ
        # ----------------------------------------------------

        seed=SEED,
    )


# ============================================================
# CONSTRUCTION DU TRAINER
# ============================================================

def build_trainer(
    model,
    tokenizer,
    dataset,
    output_dir: str = "artifacts/checkpoints",
):
    """
    Construit le SFTTrainer utilisé pour le fine-tuning.

    Parameters
    ----------
    model
        Modèle Qwen quantifié en 4 bits et préparé avec
        prepare_model_for_kbit_training().

    tokenizer
        Tokenizer Qwen.

    dataset
        DatasetDict contenant :
        - train
        - validation
        - test

    output_dir : str
        Emplacement des checkpoints.

    Returns
    -------
    SFTTrainer
        Trainer prêt à être lancé avec :

            trainer.train()
    """

    # --------------------------------------------------------
    # CONFIGURATION LoRA
    # --------------------------------------------------------

    lora_config = build_lora_config()


    # --------------------------------------------------------
    # PARAMÈTRES D'ENTRAÎNEMENT
    # --------------------------------------------------------

    training_args = build_training_args(
        output_dir=output_dir,
    )


    # --------------------------------------------------------
    # CONSTRUCTION DU TRAINER
    # --------------------------------------------------------

    trainer = SFTTrainer(

        # Qwen quantifié en 4 bits.
        model=model,


        # Hyperparamètres.
        args=training_args,


        # Données utilisées pour modifier les adaptateurs LoRA.
        train_dataset=dataset["train"],


        # Données utilisées pour mesurer la généralisation
        # pendant l'entraînement.
        eval_dataset=dataset["validation"],


        # Tokenizer utilisé pour transformer les conversations
        # en séquences de tokens.
        processing_class=tokenizer,


        # Cette configuration indique au SFTTrainer d'injecter
        # les adaptateurs LoRA dans le modèle.
        peft_config=lora_config,
    )

    return trainer