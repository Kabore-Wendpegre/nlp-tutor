# 🎓 NLP Tutor

**NLP Tutor** est un assistant pédagogique francophone spécialisé dans le **traitement automatique du langage naturel (NLP)**, les **modèles de langage** et les **Transformers**.

Le projet repose sur le modèle :

**Qwen/Qwen2.5-1.5B-Instruct**

adapté avec une méthode de fine-tuning efficace en mémoire :

**QLoRA — Quantized Low-Rank Adaptation**

L'objectif du projet est de comparer un modèle de base guidé uniquement par prompting à une version spécialisée avec QLoRA, puis d'intégrer le modèle obtenu dans une interface conversationnelle **Gradio**.

---

# 🚀 Tester directement NLP Tutor

## Méthode recommandée : Google Colab avec GPU

L'application utilise un modèle Qwen quantifié en 4 bits avec `bitsandbytes`.

L'utilisation d'un **GPU CUDA** est donc recommandée.

Google Colab permet de lancer l'application sans avoir besoin d'un ordinateur puissant.

> Aucun réentraînement n'est nécessaire pour utiliser l'application.
>
> L'adaptateur LoRA entraîné est déjà fourni dans le dépôt.

---

## 1. Ouvrir Google Colab

Ouvrir :

https://colab.research.google.com/

Créer un nouveau notebook.

---

## 2. Activer le GPU

Dans Google Colab :

**Exécution → Modifier le type d'exécution → Accélérateur matériel → GPU**

Il est possible de vérifier ensuite que le GPU est disponible avec :

```python
import torch

print("CUDA disponible :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
```

Un GPU de type **Tesla T4** est suffisant pour lancer NLP Tutor.

---

## 3. Cloner le dépôt GitHub

Dans une première cellule Colab :

```python
!git clone https://github.com/Kabore-Wendpegre/nlp-tutor.git
```

---

## 4. Se placer dans le dossier du projet

```python
%cd /content/nlp-tutor
```

La structure est actuellement :

```text
nlp-tutor/
│
├── app/
├── data/
├── lora_adapter/
├── notebooks/
├── results/
├── src/
├── Presentation_NLP_Tutor.pdf
├── Presentation_NLP_Tutor_.pptx
├── Rapport_NLP_Tutor_style_RMarkdown (1).pdf
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 5. Installer le projet et les dépendances GPU

```python
%pip install -q -e ".[gpu]"
```

Cette commande installe notamment les bibliothèques nécessaires au projet :

- PyTorch
- Transformers
- PEFT
- Accelerate
- Datasets
- TRL
- bitsandbytes
- Gradio
- rouge-score

---

## 6. Lancer l'application

```python
%run app/app.py
```

Au démarrage, l'application :

1. vérifie la présence d'un GPU CUDA ;
2. charge le tokenizer Qwen ;
3. télécharge le modèle de base `Qwen2.5-1.5B-Instruct` depuis Hugging Face ;
4. charge l'adaptateur LoRA présent dans `lora_adapter/` ;
5. lance l'interface Gradio.

Un lien public Gradio sera ensuite affiché, par exemple :

```text
Running on public URL:
https://xxxxxxxxxxxxxxxx.gradio.live
```

Il suffit d'ouvrir ce lien dans le navigateur pour utiliser **NLP Tutor**.

---

# 💬 Exemples de questions

L'interface permet notamment de tester les questions suivantes :

```text
Qu'est-ce qu'un token en NLP ?
```

```text
Explique-moi simplement la self-attention.
```

```text
Quelle est la différence entre LoRA et QLoRA ?
```

```text
À quoi sert la température pendant la génération ?
```

```text
Quelle différence y a-t-il entre les logits et les probabilités ?
```

```text
Pourquoi utilise-t-on un jeu de validation ?
```

---

# 🧠 Fonctionnement de l'application

L'application finale n'exécute **aucun entraînement**.

Le fonctionnement est :

```text
Question utilisateur
        │
        ▼
   Interface Gradio
        │
        ▼
   Tokenizer Qwen
        │
        ▼
Qwen2.5-1.5B-Instruct
        +
 Adaptateur LoRA
        │
        ▼
 Génération de texte
        │
        ▼
 Réponse pédagogique
```

Le modèle est chargé une seule fois au démarrage de l'application.

Les questions suivantes réutilisent ensuite le même modèle en mémoire.

---

# 🖥️ Exécution locale

L'application peut également être lancée sur une machine disposant d'un **GPU NVIDIA compatible CUDA**.

Après avoir cloné le dépôt :

```bash
git clone https://github.com/Kabore-Wendpegre/nlp-tutor.git
```

Puis :

```bash
cd nlp-tutor/nlp_tutor
```

Créer éventuellement un environnement virtuel :

```bash
python3 -m venv .venv
```

L'activer sous Linux/macOS :

```bash
source .venv/bin/activate
```

Puis installer le projet :

```bash
python3 -m pip install -e ".[gpu]"
```

Enfin :

```bash
python3 app/app.py
```

> L'exécution locale sans GPU CUDA n'est pas la méthode recommandée.
>
> Pour une reproduction simple et fiable, utiliser **Google Colab avec GPU**.

---

# 📓 Reproduire l'expérience complète

Le notebook principal du projet est :

```text
notebooks/agent.ipynb
```

Contrairement à `app/app.py`, ce notebook contient l'ensemble de l'expérience.

Il permet de reproduire :

1. la vérification de l'environnement ;
2. la tokenisation ;
3. l'étude des logits ;
4. l'application de la softmax ;
5. les stratégies de décodage ;
6. la baseline par prompting ;
7. la construction du dataset ;
8. le fine-tuning QLoRA ;
9. la validation ;
10. l'évaluation ;
11. la comparaison baseline / QLoRA ;
12. les tests complémentaires de généralisation.

Pour simplement tester l'application finale, **il n'est pas nécessaire d'exécuter ce notebook**.

---

# 🎯 Objectif du projet

Le projet cherche à répondre à la question suivante :

> Une adaptation légère avec QLoRA permet-elle d'améliorer le comportement pédagogique de Qwen2.5-1.5B-Instruct par rapport à une simple stratégie de prompting ?

Le système cible principalement des étudiants souhaitant réviser des notions fondamentales de NLP.

Les réponses attendues doivent :

- être en français ;
- commencer par une définition claire ;
- utiliser des explications progressives ;
- fournir un exemple lorsque cela est pertinent ;
- terminer par une synthèse de type **« À retenir : ... »**.

---

# 🤖 Modèle utilisé

Modèle de base :

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Le modèle possède environ :

```text
1,56 milliard de paramètres
```

Au lieu d'effectuer un fine-tuning complet, le projet utilise **QLoRA**.

---

# 🔧 QLoRA

QLoRA combine deux mécanismes.

## Quantification

Le modèle de base est chargé en **4 bits** afin de réduire la consommation de mémoire GPU.

Configuration utilisée :

```python
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_use_double_quant=True
bnb_4bit_compute_dtype=torch.float16
```

## LoRA

Les poids principaux du modèle restent gelés.

De petites matrices entraînables sont ajoutées à certaines couches du Transformer.

Configuration principale :

```text
r = 16
lora_alpha = 32
lora_dropout = 0.05
```

Modules ciblés :

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Environ :

```text
18 464 768 paramètres
```

sont entraînables sur environ :

```text
1 562 179 072 paramètres
```

soit environ :

```text
1,182 %
```

des paramètres du modèle.

---

# 📚 Dataset

Le dataset synthétique contient :

```text
500 exemples
```

répartis sur :

```text
25 concepts NLP
```

avec :

```text
20 formulations par concept
```

La répartition utilisée est :

| Ensemble | Nombre d'exemples |
|---|---:|
| Entraînement | 350 |
| Validation | 75 |
| Test | 75 |
| **Total** | **500** |

Les concepts incluent notamment :

- token ;
- tokenizer ;
- vocabulaire ;
- embedding ;
- logits ;
- softmax ;
- température ;
- top-k ;
- top-p ;
- attention ;
- self-attention ;
- Query / Key / Value ;
- Transformer ;
- encodage positionnel ;
- masque causal ;
- pré-entraînement ;
- fine-tuning ;
- instruction tuning ;
- LoRA ;
- QLoRA ;
- quantification ;
- surapprentissage ;
- train / validation / test.

---

# ⚙️ Configuration de l'entraînement

Configuration principale :

```text
Epochs                     : 2
Batch size par GPU         : 2
Gradient accumulation      : 8
Batch effectif             : 16
Learning rate              : 2e-4
Longueur maximale          : 512 tokens
Optimiseur                 : AdamW
Gradient checkpointing     : activé
```

GPU utilisé pendant l'expérience :

```text
NVIDIA Tesla T4
```

---

# 📊 Évaluation

Deux métriques automatiques principales sont utilisées.

## ROUGE-L

ROUGE-L mesure la proximité séquentielle entre une réponse générée et une réponse de référence.

## Keyword Coverage

La couverture des mots-clés mesure la proportion de notions importantes présentes dans la réponse.

---

# 📈 Résultats principaux

| Configuration | ROUGE-L | Keyword Coverage |
|---|---:|---:|
| Baseline Qwen | 0.1331 | 0.2500 |
| QLoRA | 1.0000 | 0.8375 |

L'adaptation QLoRA améliore fortement les performances sur le jeu de test utilisé.

Cependant, le score ROUGE-L très élevé doit être interprété avec prudence.

Les réponses de référence du dataset sont relativement standardisées entre les différentes reformulations d'un même concept.

Un score élevé ne signifie donc pas que le modèle généralise parfaitement à toutes les questions NLP.

---

# 🧪 Tests complémentaires

Un ensemble supplémentaire de questions plus éloignées des formulations utilisées pendant l'entraînement a été ajouté.

Par exemple :

```text
Pourquoi un même mot peut-il être tokenisé différemment selon le tokenizer ?
```

```text
Quelle différence y a-t-il entre les logits et les probabilités finales ?
```

```text
Dans quel cas top-p peut-il être préférable à top-k ?
```

```text
Pourquoi un Transformer a-t-il besoin d'informations de position ?
```

Ces questions permettent d'étudier davantage la capacité de généralisation du modèle.

---

# 📁 Structure du projet

```text
nlp_tutor/
│
├── app/
│   └── app.py
│
├── data/
│
├── lora_adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── ...
│
├── notebooks/
│   └── agent.ipynb
│
├── results/
│   ├── baseline_results.csv
│   ├── baseline_vs_qlora.csv
│   ├── challenge_comparison.csv
│   ├── challenge_examples_inspection.csv
│   ├── final_model_tests.csv
│   ├── metrics_summary.csv
│   └── qlora_results.csv
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
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# 🧩 Rôle des principaux fichiers

### `config.py`

Contient notamment :

- le modèle utilisé ;
- la seed ;
- le prompt système.

### `model.py`

Gère :

- le tokenizer ;
- le modèle de base ;
- le chargement quantifié ;
- le modèle QLoRA ;
- le rechargement de l'adaptateur.

### `generation.py`

Contient :

- le chat template ;
- la génération ;
- les stratégies de décodage.

### `dataset.py`

Construit :

- les concepts ;
- les formulations ;
- les exemples ;
- les splits train / validation / test.

### `training.py`

Contient :

- la configuration bitsandbytes ;
- la configuration LoRA ;
- la configuration SFT ;
- le trainer.

### `evaluation.py`

Contient :

- la normalisation des réponses ;
- ROUGE-L ;
- keyword coverage ;
- la sauvegarde des résultats.

### `app/app.py`

Charge le modèle final et lance l'interface **Gradio**.

### `notebooks/agent.ipynb`

Orchestre et documente l'ensemble de l'expérience.

---

# 💾 Adaptateur entraîné

Le fine-tuning produit un adaptateur LoRA sauvegardé dans :

```text
lora_adapter/
```

Le fichier principal est :

```text
adapter_model.safetensors
```

Le modèle complet Qwen n'est pas stocké dans ce dépôt.

Il est téléchargé automatiquement depuis Hugging Face au premier lancement.

L'application combine donc :

```text
Qwen2.5-1.5B-Instruct
          +
   adaptateur LoRA
          ↓
      NLP Tutor
```

---

# ⚠️ Limites actuelles

Le projet présente plusieurs limites :

- dataset synthétique relativement petit ;
- réponses de référence assez standardisées ;
- ROUGE-L favorisé par la structure du dataset ;
- certaines questions de raisonnement ou de comparaison restent difficiles ;
- l'historique Gradio est reçu par l'application mais n'est pas encore réellement réinjecté dans le contexte du modèle ;
- l'évaluation repose principalement sur des métriques automatiques.

---

# 🔮 Améliorations possibles

Parmi les évolutions envisageables :

- augmenter la diversité des données ;
- ajouter davantage de réponses de référence ;
- introduire des questions hors périmètre ;
- tester davantage de questions de raisonnement ;
- intégrer réellement l'historique conversationnel ;
- ajouter une évaluation humaine ;
- tester plusieurs modèles ;
- comparer plusieurs configurations LoRA ;
- déployer l'application de manière permanente.

---

# 🛠️ Technologies utilisées

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- QLoRA
- bitsandbytes
- TRL
- Datasets
- Gradio
- Qwen2.5-1.5B-Instruct
- Google Colab
- Git / GitHub

---

# ✅ Résumé pour tester rapidement

Sur un **Google Colab avec GPU**, les seules commandes nécessaires sont :

```python
!git clone https://github.com/Kabore-Wendpegre/nlp-tutor.git
```

```python
%cd /content/nlp-tutor
```

```python
%pip install -q -e ".[gpu]"
```

```python
%run app/app.py
```

Puis ouvrir le lien :

```text
https://xxxxxxxx.gradio.live
```

généré par Gradio.

**Il n'est pas nécessaire de relancer l'entraînement pour utiliser NLP Tutor.**