# NLP Final Project — Domain Shift in Sentiment Classification

**Research Question:** How do different NLP models vary in their ability to generalize from general-domain sentiment data to financial-domain sentiment classification under domain shift?

## Project Structure

```
nlp_project/
├── data_loader.py      # Dataset loading & preprocessing (IMDB + Financial PhraseBank)
├── features.py         # TF-IDF feature extraction
├── models.py           # Classical model training (Naive Bayes, Logistic Regression)
├── evaluate.py         # Metrics, confusion matrices, comparison plots
├── run_classical.py    # Main experiment runner (3 experiments)
├── requirements.txt
└── outputs/            # Generated results, plots, and tables
```

## Module Overview

| Module | Responsibility |
|---|---|
| `data_loader.py` | Loads IMDB (50K reviews) and Financial PhraseBank via HuggingFace `datasets`. Cleans text, removes neutral FPB examples, provides train/test splits. |
| `features.py` | TF-IDF vectorization with configurable n-grams, max features, and sublinear TF. Supports fit/transform and transform-only (for cross-domain). |
| `models.py` | Trains Multinomial NB, Complement NB, and Logistic Regression with various hyperparameter configs (C values, class weighting). |
| `evaluate.py` | Computes accuracy/precision/recall/F1, generates confusion matrix heatmaps, builds comparison tables, and produces cross-experiment bar plots. |
| `run_classical.py` | Orchestrates the 3 experiments and generates summary visualizations. |

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Run all 3 experiments + summary plots
python run_classical.py

# Run a specific experiment
python run_classical.py --experiment 1   # In-domain: IMDB → IMDB
python run_classical.py --experiment 2   # Cross-domain: IMDB → Financial PhraseBank
python run_classical.py --experiment 3   # Fine-tuned: FPB → FPB
```

## Experiments

1. **Experiment 1 — In-Domain (IMDB → IMDB):** Baseline performance on general sentiment.
2. **Experiment 2 — Cross-Domain (IMDB → Financial PhraseBank):** Measures domain shift degradation.
3. **Experiment 3 — Fine-Tuned (FPB → FPB):** In-domain financial performance to compare against cross-domain.

## Classical Model Variants Tested

- **Multinomial Naive Bayes** (alpha=1.0)
- **Complement Naive Bayes** (alpha=1.0)
- **Logistic Regression** — C ∈ {0.1, 1.0, 10.0}, with and without balanced class weights

## Outputs

After running, `outputs/` will contain:
- `exp{1,2,3}_comparison.csv` — per-experiment metrics tables
- `exp{1,2,3}_cm_*.png` — confusion matrices for each model/experiment
- `exp{1,2,3}_results.json` — raw metrics in JSON
- `summary_all_experiments.csv` — combined F1 comparison + domain shift drop
- `summary_performance_drop.png` — bar chart of in-domain vs cross-domain F1
- `summary_finetune_improvement.png` — bar chart of cross-domain vs fine-tuned F1
