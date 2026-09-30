"""
Évaluation des réponses produites par NLP Tutor.

Ce module regroupe :

1. la normalisation du texte ;
2. le calcul de la couverture des mots-clés ;
3. le calcul de ROUGE-L ;
4. l'évaluation d'un ensemble de questions ;
5. la sauvegarde des résultats dans un fichier CSV.

Il ne charge pas le modèle et ne construit pas le dataset.
"""

from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

from rouge_score import rouge_scorer

from .generation import generate_tutor


# ============================================================
# CONFIGURATION DE ROUGE
# ============================================================

# Le scorer est créé une seule fois.
#
# ROUGE-L compare la plus longue sous-séquence commune
# entre la réponse de référence et la réponse générée.
#
# Exemple :
#
# référence :
#   "un token est une unité de texte"
#
# prédiction :
#   "un token correspond à une unité de texte"
#
# Même si les phrases ne sont pas exactement identiques,
# ROUGE-L peut détecter une partie importante de structure commune.
ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=False,
)


# ============================================================
# NORMALISATION DU TEXTE
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalise un texte afin de faciliter les comparaisons.

    Les opérations réalisées sont :

    1. passage en minuscules ;
    2. suppression des accents ;
    3. suppression des espaces multiples.

    Exemple
    -------
    "ÉCHANTILLONNAGE  Probabilité"
            ↓
    "echantillonnage probabilite"

    Parameters
    ----------
    text : str
        Texte à normaliser.

    Returns
    -------
    str
        Texte normalisé.
    """

    # Passage en minuscules.
    text = text.lower()

    # Décomposition Unicode.
    #
    # Exemple :
    #
    # é
    #
    # devient conceptuellement :
    #
    # e + accent
    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    # On retire les caractères représentant les accents.
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    # split() sans argument supprime automatiquement
    # les espaces multiples, tabulations, retours ligne, etc.
    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# COUVERTURE DES MOTS-CLÉS
# ============================================================

def keyword_coverage(
    prediction: str,
    keywords: list[str],
) -> float:
    """
    Calcule la proportion de mots-clés présents dans une réponse.

    Formule :

        nombre de mots-clés trouvés
        ---------------------------
        nombre total de mots-clés

    Exemple
    -------

    keywords = [
        "softmax",
        "probabilités",
        "logits",
        "distribution"
    ]

    Si la réponse contient 3 mots-clés sur 4 :

        coverage = 3 / 4 = 0.75

    Parameters
    ----------
    prediction : str
        Réponse générée par le modèle.

    keywords : list[str]
        Liste de termes importants attendus.

    Returns
    -------
    float
        Score compris entre 0 et 1.
    """

    # Normalisation de la réponse.
    prediction_normalized = normalize_text(
        prediction
    )

    # Normalisation de chaque mot-clé.
    normalized_keywords = [
        normalize_text(keyword)
        for keyword in keywords
    ]

    # Cas particulier :
    # aucun mot-clé défini.
    if not normalized_keywords:
        return np.nan

    # Nombre de mots-clés présents dans la réponse.
    hits = sum(
        keyword in prediction_normalized
        for keyword in normalized_keywords
    )

    # Proportion de mots-clés retrouvés.
    return hits / len(normalized_keywords)


# ============================================================
# ROUGE-L
# ============================================================

def compute_rouge_l(
    reference: str,
    prediction: str,
) -> float:
    """
    Calcule le ROUGE-L F1 entre une référence et une prédiction.

    Parameters
    ----------
    reference : str
        Réponse attendue.

    prediction : str
        Réponse produite par le modèle.

    Returns
    -------
    float
        ROUGE-L F1 compris entre 0 et 1.
    """

    scores = ROUGE_SCORER.score(
        reference,
        prediction,
    )

    return scores["rougeL"].fmeasure


# ============================================================
# ÉVALUATION D'UN DATASET
# ============================================================

def evaluate_dataset(
    model,
    tokenizer,
    eval_dataset,
    model_name: str,
    max_new_tokens: int = 220,
) -> pd.DataFrame:
    """
    Évalue un modèle sur plusieurs exemples.

    Pour chaque question :

        question
           ↓
        génération
           ↓
        ROUGE-L
           +
        couverture mots-clés

    Une ligne du DataFrame correspond donc à une question.

    Parameters
    ----------
    model
        Modèle à évaluer.

    tokenizer
        Tokenizer associé.

    eval_dataset
        Dataset Hugging Face contenant au minimum :

        - concept
        - question
        - reference
        - keywords

    model_name : str
        Nom utilisé dans le tableau final.

        Exemple :
        "Qwen2.5-1.5B + prompting"

    max_new_tokens : int
        Taille maximale d'une réponse.

    Returns
    -------
    pandas.DataFrame
        Résultats détaillés.
    """

    results = []

    # enumerate(..., start=1)
    #
    # permet d'avoir :
    #
    # 1, 2, 3...
    #
    # au lieu de :
    #
    # 0, 1, 2...
    for index, item in enumerate(
        eval_dataset,
        start=1,
    ):

        # ----------------------------------------------------
        # 1. GÉNÉRATION
        # ----------------------------------------------------

        # do_sample=False :
        #
        # génération déterministe.
        #
        # C'est très important pendant l'évaluation,
        # car nous voulons comparer les modèles sans ajouter
        # de hasard inutile.
        prediction = generate_tutor(
            question=item["question"],
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )


        # ----------------------------------------------------
        # 2. ROUGE-L
        # ----------------------------------------------------

        rouge_l = compute_rouge_l(
            reference=item["reference"],
            prediction=prediction,
        )


        # ----------------------------------------------------
        # 3. COUVERTURE DES MOTS-CLÉS
        # ----------------------------------------------------

        coverage = keyword_coverage(
            prediction=prediction,
            keywords=item["keywords"],
        )


        # ----------------------------------------------------
        # 4. SAUVEGARDE DE LA LIGNE
        # ----------------------------------------------------

        results.append({
            "id": index,

            "modele": model_name,

            "concept": item["concept"],

            "question": item["question"],

            "reference": item["reference"],

            "prediction": prediction,

            "rougeL_f1": rouge_l,

            "keyword_coverage": coverage,
        })


    # Conversion de la liste de dictionnaires en DataFrame.
    return pd.DataFrame(results)


# ============================================================
# SYNTHÈSE DES RÉSULTATS
# ============================================================

def summarize_results(
    results: pd.DataFrame,
) -> dict:
    """
    Calcule les métriques moyennes d'une évaluation.

    Parameters
    ----------
    results : pandas.DataFrame
        Résultat produit par evaluate_dataset().

    Returns
    -------
    dict
        Exemple :

        {
            "nombre_questions": 20,
            "rougeL_moyen": 0.1331,
            "keyword_coverage_moyenne": 0.25
        }
    """

    return {
        "nombre_questions": len(results),

        "rougeL_moyen":
            results["rougeL_f1"].mean(),

        "keyword_coverage_moyenne":
            results["keyword_coverage"].mean(),
    }


# ============================================================
# SAUVEGARDE DES RÉSULTATS
# ============================================================

def save_results(
    results: pd.DataFrame,
    path,
):
    """
    Sauvegarde un DataFrame d'évaluation au format CSV.

    Exemple :

        save_results(
            baseline_results,
            "results/baseline_results.csv"
        )

    Le dossier parent est créé automatiquement s'il n'existe pas.
    """

    path = Path(path)

    # Exemple :
    #
    # results/baseline_results.csv
    #
    # path.parent donne :
    #
    # results/
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # index=False évite d'ajouter une colonne pandas
    # inutile dans le fichier CSV.
    results.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )

    print(
        f"Résultats sauvegardés : {path}"
    )