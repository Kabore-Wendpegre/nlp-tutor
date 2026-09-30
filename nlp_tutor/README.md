# NLP Tutor

Assistant pédagogique francophone spécialisé dans le **Natural Language Processing (NLP)**, les **modèles de langage** et les **Transformers**.

Le projet a pour objectif de comprendre, construire, adapter et évaluer un petit modèle de langage en étudiant progressivement les principales notions vues en cours :

- tokenisation ;
- logits ;
- softmax ;
- argmax ;
- température ;
- top-k ;
- top-p ;
- prompting ;
- séparation train / validation / test ;
- quantification ;
- LoRA ;
- QLoRA ;
- fine-tuning ;
- évaluation ;
- déploiement avec Gradio.

---

# 1. Objectif du projet

L'objectif est de construire un assistant pédagogique capable de répondre en français à des questions portant sur le NLP et les Transformers.

Deux approches sont comparées :

### Baseline

Le modèle de base est utilisé avec un **system prompt**, sans modifier ses paramètres.

### QLoRA

Le même modèle est ensuite adapté sur un dataset pédagogique grâce à **QLoRA**.

Cette comparaison permet d'étudier concrètement la différence entre :

```text
prompting
        ↓
modèle pré-entraîné

et

dataset spécialisé
        ↓
QLoRA
        ↓
modèle adapté
```

---

# 2. Modèle utilisé

Le modèle utilisé dans le projet est :

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Il s'agit d'un modèle causal de type Transformer.

Pour le fine-tuning, le modèle est chargé en **quantification 4 bits** puis adapté grâce à des matrices LoRA.

---

# 3. Architecture du projet

Le projet n'est pas organisé comme un unique notebook.

Le code réutilisable est séparé du notebook expérimental et de l'application.

```text
nlp_tutor/
│
├── app/
│   └── app.py
│
├── artifacts/
│   └── lora_adapter/
│
├── data/
│
├── notebooks/
│   ├── agent.ipynb
│   └── agent_legacy.ipynb
│
├── results/
│
├── src/
│   └── nlp_tutor/
│       ├── __init__.py
│       ├── config.py
│       ├── dataset.py
│       ├── evaluation.py
│       ├── generation.py
│       ├── model.py
│       └── training.py
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# 4. Rôle des différents modules

## `config.py`

Contient la configuration générale du projet :

- identifiant du modèle ;
- seed ;
- paramètres de génération ;
- prompt système ;
- constantes utilisées dans plusieurs modules.

---

## `model.py`

Centralise le chargement des modèles.

Il permet notamment de charger :

- le tokenizer ;
- le modèle de base ;
- le modèle quantifié pour QLoRA ;
- le modèle final avec son adaptateur LoRA.

Cela évite de répéter la logique de chargement dans plusieurs fichiers.

---

## `generation.py`

Contient les fonctions responsables de la génération de texte.

Il gère notamment :

- greedy decoding ;
- température ;
- top-k ;
- top-p ;
- génération de réponses pédagogiques.

---

## `dataset.py`

Construit le dataset pédagogique utilisé pour l'adaptation du modèle.

Le dataset est séparé en :

```text
train
validation
test
```

Cette séparation permet d'éviter d'évaluer directement le modèle sur les exemples utilisés pour ajuster ses paramètres.

---

## `training.py`

Contient la logique d'entraînement :

```text
quantification
      ↓
LoRA
      ↓
QLoRA
      ↓
SFTTrainer
```

Ce module contient notamment :

- la configuration LoRA ;
- la configuration du Trainer ;
- la construction du `SFTTrainer`.

---

## `evaluation.py`

Contient les métriques et fonctions utilisées pour comparer les modèles.

Les évaluations comprennent notamment :

- ROUGE-L ;
- couverture de mots-clés ;
- comparaison baseline / QLoRA ;
- analyse qualitative sur des questions nouvelles.

---

## `notebooks/agent.ipynb`

Notebook principal de démonstration.

Il est utilisé pour :

1. comprendre la tokenisation ;
2. observer les logits ;
3. appliquer softmax ;
4. étudier argmax ;
5. comparer différentes stratégies de décodage ;
6. tester le prompting ;
7. construire le dataset ;
8. entraîner l'adaptateur QLoRA ;
9. comparer les performances ;
10. analyser la généralisation.

Le notebook sert donc principalement à **l'expérimentation et à l'explication pédagogique**.

Le code métier réutilisable se trouve dans `src/nlp_tutor/`.

---

## `notebooks/agent_legacy.ipynb`

Ancienne version du notebook conservée pour garder une trace du travail initial.

Elle n'est plus utilisée comme notebook principal.

---

## `app/app.py`

Application finale basée sur Gradio.

Elle :

1. charge le tokenizer ;
2. charge le modèle quantifié ;
3. recharge l'adaptateur LoRA ;
4. reçoit une question utilisateur ;
5. génère une réponse pédagogique.

L'entraînement n'est pas effectué dans l'application.

---

# 5. Pipeline étudié

Le projet couvre progressivement le pipeline suivant :

```text
Texte utilisateur
        ↓
Tokenisation
        ↓
Token IDs
        ↓
Transformer
        ↓
Logits
        ↓
Softmax
        ↓
Distribution de probabilités
        ↓
Décodage
 ┌──────┼────────┬───────┐
greedy température top-k top-p
        ↓
Réponse
```

La deuxième partie du projet ajoute :

```text
Dataset pédagogique
        ↓
Train / Validation / Test
        ↓
Modèle quantifié 4 bits
        ↓
LoRA
        ↓
QLoRA
        ↓
Fine-tuning
        ↓
Adaptateur LoRA
        ↓
Évaluation
        ↓
Application Gradio
```

---

# 6. Tokenisation

Un modèle de langage ne travaille pas directement avec les mots.

Le texte est d'abord transformé en **tokens**, puis chaque token est associé à un identifiant numérique.

Exemple conceptuel :

```text
"Un modèle de langage"

        ↓ tokenizer

["Un", " modèle", " de", " langage"]

        ↓

[1806, 82497, 409, ...]
```

La segmentation dépend du tokenizer utilisé.

---

# 7. Logits et probabilités

Pour chaque position, le modèle produit un ensemble de scores appelés **logits**.

Les logits ne sont pas directement des probabilités.

La fonction :

```text
softmax
```

transforme les logits en une distribution de probabilités :

\[
P_i =
\frac{\exp(z_i)}
{\sum_j \exp(z_j)}
\]

Le token suivant peut ensuite être sélectionné à partir de cette distribution.

---

# 8. Température

La température modifie la distribution avant l'échantillonnage :

\[
P_i =
\text{softmax}
\left(
\frac{z_i}{T}
\right)
\]

Une température faible concentre davantage la distribution.

Une température plus élevée augmente la diversité des candidats.

---

# 9. Stratégies de décodage

Plusieurs stratégies sont étudiées.

### Greedy decoding

Sélectionne systématiquement le token ayant la probabilité maximale.

### Température

Modifie la concentration de la distribution.

### Top-k

Limite l'échantillonnage aux `k` meilleurs candidats.

### Top-p

Conserve le plus petit ensemble de tokens dont la probabilité cumulée atteint un seuil donné.

---

# 10. Baseline par prompting

Avant tout fine-tuning, le modèle est testé uniquement avec un prompt système.

Le prompt lui demande notamment de :

- répondre en français ;
- commencer par une définition ;
- expliquer progressivement ;
- utiliser des exemples ;
- terminer par un élément à retenir.

Cette configuration constitue la **baseline**.

Elle permet de mesurer ce que le modèle peut déjà réaliser sans modifier ses paramètres.

---

# 11. Fine-tuning avec QLoRA

L'entraînement complet d'un modèle de plus d'un milliard de paramètres peut nécessiter une quantité importante de mémoire GPU.

Le projet utilise donc QLoRA.

Le principe est :

```text
Modèle pré-entraîné
        ↓
Quantification 4 bits
        ↓
Poids principaux gelés
        ↓
Ajout de matrices LoRA
        ↓
Entraînement uniquement des adaptateurs
```

Cette approche permet de réduire fortement la quantité de paramètres entraînables.

---

# 12. Évaluation

L'évaluation ne repose pas sur une seule métrique.

Plusieurs niveaux sont utilisés.

## Évaluation quantitative

### ROUGE-L

Compare les séquences entre la réponse produite et la réponse de référence.

### Couverture des mots-clés

Mesure la présence des notions considérées comme importantes dans la réponse.

---

# 13. Résultats expérimentaux observés

Lors de l'expérience réalisée sur le jeu de test proche du dataset :

| Configuration | ROUGE-L moyen | Couverture moyenne |
|---|---:|---:|
| Baseline + prompting | 0.1331 | 0.2500 |
| QLoRA | 1.0000 | 0.8375 |

Ces résultats montrent une très forte amélioration sur ce jeu d'évaluation.

Cependant, ils doivent être interprétés avec prudence.

Un ROUGE-L très élevé sur des questions proches des données utilisées pour l'entraînement ne prouve pas à lui seul que le modèle généralise parfaitement.

---

# 14. Évaluation sur des questions inédites

Une seconde expérience utilise des questions différentes de celles du dataset.

Elle permet d'étudier davantage la capacité de généralisation du modèle.

L'analyse qualitative réalisée sur 20 questions avait donné :

```text
QLoRA préférable : 12 / 20
Baseline préférable : 3 / 20
Résultat mitigé : 4 / 20
Aucune satisfaisante : 1 / 20
```

Cette évaluation est particulièrement importante car elle permet de distinguer :

```text
mémorisation
      ≠
généralisation
```

---

# 15. Limites de l'évaluation

Les métriques lexicales comme ROUGE-L sont utiles mais insuffisantes pour évaluer seules un assistant pédagogique.

Deux réponses peuvent transmettre la même information avec des formulations différentes.

Inversement, une réponse peut ressembler fortement à la référence tout en présentant des problèmes de raisonnement ou de généralisation.

Une évaluation plus complète peut donc combiner :

```text
métriques automatiques
        +
analyse humaine
        +
jeu de challenge
        +
évaluation spécialisée
```

Une évolution possible du projet est l'utilisation d'un framework comme **Inspect AI** pour rendre les évaluations plus structurées et reproductibles.

---

# 16. Installation

Créer un environnement Python :

```bash
python -m venv .venv
```

L'activer sous Linux :

```bash
source .venv/bin/activate
```

Puis installer le projet :

```bash
pip install -e .
```

Pour l'environnement GPU utilisé pour QLoRA :

```bash
pip install -e ".[gpu]"
```

---

# 17. Vérification du package

Tester les imports :

```bash
python -c "
from nlp_tutor.model import load_tokenizer
from nlp_tutor.dataset import build_dataset
from nlp_tutor.training import build_lora_config

print('NLP Tutor : OK')
"
```

---

# 18. Utilisation du notebook

Le notebook principal se trouve dans :

```text
notebooks/agent.ipynb
```

Il doit être exécuté avec un environnement GPU pour les sections nécessitant QLoRA.

L'environnement utilisé pendant le développement était notamment :

```text
Google Colab
GPU NVIDIA Tesla T4
```

---

# 19. Adaptateur LoRA

Après l'entraînement, l'adaptateur est sauvegardé dans :

```text
artifacts/lora_adapter/
```

Il contient uniquement les paramètres supplémentaires nécessaires à l'adaptation du modèle.

Le modèle de base n'est donc pas recopié intégralement dans le dépôt.

---

# 20. Lancement de l'application

Lorsque l'adaptateur est disponible et qu'un GPU CUDA est accessible :

```bash
python app/app.py
```

L'application lance alors une interface Gradio permettant de poser des questions à NLP Tutor.

Exemples :

```text
Qu'est-ce qu'un token ?

Explique-moi la self-attention.

Quelle différence entre LoRA et QLoRA ?

Pourquoi applique-t-on softmax aux logits ?

À quoi sert la température ?
```

---

# 21. Technologies utilisées

```text
Python
PyTorch
Transformers
Hugging Face
Datasets
PEFT
LoRA
QLoRA
bitsandbytes
TRL
ROUGE
Pandas
Gradio
Jupyter
Google Colab
```

---

# 22. Améliorations possibles

Plusieurs extensions sont envisageables :

- agrandir le dataset pédagogique ;
- ajouter davantage de concepts NLP ;
- tester plusieurs tailles de modèles ;
- comparer LoRA et QLoRA ;
- étudier plusieurs niveaux de quantification ;
- ajouter des évaluations sémantiques ;
- utiliser Inspect AI ;
- ajouter un LLM-as-a-judge ;
- construire un vrai jeu de benchmark indépendant ;
- gérer l'historique conversationnel ;
- déployer l'application.

---

# 23. Conclusion

Ce projet met en œuvre l'ensemble de la chaîne permettant de passer d'un modèle de langage pré-entraîné à un assistant spécialisé :

```text
Comprendre le Transformer
            ↓
Construire une baseline
            ↓
Créer un dataset
            ↓
Fine-tuner avec QLoRA
            ↓
Évaluer
            ↓
Analyser la généralisation
            ↓
Construire une application
```

L'objectif n'est donc pas uniquement d'obtenir un chatbot fonctionnel, mais de comprendre expérimentalement les mécanismes essentiels utilisés dans les modèles de langage modernes.