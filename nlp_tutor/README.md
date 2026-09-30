# NLP Tutor

NLP Tutor est un assistant pédagogique francophone spécialisé en NLP,
modèles de langage et Transformers.

Le projet utilise **Qwen2.5-1.5B-Instruct**, adapté avec **QLoRA**.

L'application finale utilise **Gradio** pour permettre à l'utilisateur
de poser directement des questions au modèle.

---

## 🚀 Lancer l'application

### Méthode recommandée : Google Colab avec GPU

L'application utilise un modèle quantifié en 4 bits avec `bitsandbytes`.
Un environnement avec GPU CUDA est donc recommandé.

### 1. Cloner le dépôt

```bash
git clone https://github.com/Kabore-Wendpegre/nlp-tutor.git
cd nlp-tutordonc je mets just