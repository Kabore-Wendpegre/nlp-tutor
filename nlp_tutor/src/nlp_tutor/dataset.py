"""
Construction du dataset pédagogique de NLP Tutor.

Le contenu pédagogique est stocké séparément dans :

    data/concepts.json
    data/question_templates.json

Ce fichier contient uniquement la logique permettant de transformer
ces fichiers en Dataset Hugging Face.

Le dataset final possède trois splits :

    train       : 350 exemples
    validation  : 75 exemples
    test        : 75 exemples
"""

import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict

from .config import SEED, SYSTEM_PROMPT


# ============================================================
# LECTURE D'UN FICHIER JSON
# ============================================================

def load_json(path):
    """
    Charge un fichier JSON et retourne son contenu Python.

    Parameters
    ----------
    path : str ou Path
        Chemin du fichier JSON.

    Returns
    -------
    dict ou list
        Contenu du JSON.
    """

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# CRÉATION D'UN EXEMPLE
# ============================================================

def make_example(
    concept,
    question,
    answer,
    keywords,
):
    """
    Construit un exemple pédagogique complet.

    Un exemple contient :

    concept
        nom de la notion étudiée

    question
        formulation utilisateur

    reference
        réponse attendue

    keywords
        éléments importants attendus dans la réponse

    prompt
        messages fournis au modèle

    completion
        réponse cible utilisée pendant le fine-tuning
    """

    return {

        # ----------------------------------------------------
        # MÉTADONNÉES
        # ----------------------------------------------------

        "concept": concept,

        "question": question,

        "reference": answer,

        "keywords": keywords,


        # ----------------------------------------------------
        # ENTRÉE DU MODÈLE
        # ----------------------------------------------------

        "prompt": [

            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },

            {
                "role": "user",
                "content": question,
            },
        ],


        # ----------------------------------------------------
        # RÉPONSE CIBLE
        # ----------------------------------------------------

        "completion": [

            {
                "role": "assistant",
                "content": answer,
            }

        ],
    }


# ============================================================
# CONSTRUCTION DU DATASET
# ============================================================

def build_dataset(
    concepts_path="data/concepts.json",
    templates_path="data/question_templates.json",
):
    """
    Construit le DatasetDict complet.

    Pour chaque concept :

        20 formulations différentes

    puis :

        14 -> train
         3 -> validation
         3 -> test

    Avec 25 concepts :

        25 × 14 = 350 train
        25 ×  3 =  75 validation
        25 ×  3 =  75 test

    Total :

        500 exemples
    """

    # --------------------------------------------------------
    # 1. CHARGER LES DONNÉES
    # --------------------------------------------------------

    concepts = load_json(concepts_path)

    templates = load_json(templates_path)


    # --------------------------------------------------------
    # 2. VÉRIFICATIONS
    # --------------------------------------------------------

    # Notre protocole actuel repose sur exactement 25 concepts.
    if len(concepts) != 25:

        raise ValueError(
            f"25 concepts attendus, "
            f"{len(concepts)} trouvés."
        )


    # Chaque concept doit disposer de 20 reformulations.
    if len(templates) != 20:

        raise ValueError(
            f"20 templates attendus, "
            f"{len(templates)} trouvés."
        )


    # --------------------------------------------------------
    # 3. LISTES QUI RECEVRONT LES EXEMPLES
    # --------------------------------------------------------

    train_rows = []

    validation_rows = []

    test_rows = []


    # --------------------------------------------------------
    # 4. PARCOURIR LES CONCEPTS
    # --------------------------------------------------------

    for concept_index, (concept, info) in enumerate(
        concepts.items()
    ):

        # Une graine spécifique par concept.
        #
        # Exemple :
        #
        # premier concept  -> seed 42
        # deuxième         -> seed 43
        # troisième        -> seed 44
        #
        # Cela permet un mélange reproductible mais différent
        # pour chaque notion.
        rng = random.Random(
            SEED + concept_index
        )


        # On copie la liste afin de ne pas modifier
        # la liste originale.
        shuffled_templates = templates.copy()


        # Mélange reproductible.
        rng.shuffle(
            shuffled_templates
        )


        # ----------------------------------------------------
        # 5. CRÉER LES 20 QUESTIONS
        # ----------------------------------------------------

        questions = [

            template.format(
                concept=concept
            )

            for template in shuffled_templates

        ]


        # ----------------------------------------------------
        # 6. EXTRAIRE LA RÉPONSE ET LES MOTS-CLÉS
        # ----------------------------------------------------

        answer = info["answer"]

        keywords = info["keywords"]


        # ----------------------------------------------------
        # 7. CRÉER LES EXEMPLES
        # ----------------------------------------------------

        examples = [

            make_example(
                concept=concept,
                question=question,
                answer=answer,
                keywords=keywords,
            )

            for question in questions

        ]


        # ----------------------------------------------------
        # 8. SPLIT 14 / 3 / 3
        # ----------------------------------------------------

        train_rows.extend(
            examples[:14]
        )

        validation_rows.extend(
            examples[14:17]
        )

        test_rows.extend(
            examples[17:20]
        )


    # --------------------------------------------------------
    # 9. CONVERSION EN DATASETS HUGGING FACE
    # --------------------------------------------------------

    dataset = DatasetDict({

        "train":
            Dataset.from_list(train_rows),

        "validation":
            Dataset.from_list(validation_rows),

        "test":
            Dataset.from_list(test_rows),

    })


    # --------------------------------------------------------
    # 10. CONTRÔLES
    # --------------------------------------------------------

    # Ces assertions permettent de détecter immédiatement
    # une modification accidentelle du protocole.
    assert len(dataset["train"]) == 350

    assert len(dataset["validation"]) == 75

    assert len(dataset["test"]) == 75


    return dataset