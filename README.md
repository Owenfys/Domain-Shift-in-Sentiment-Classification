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

### Neural experiments (LSTM + BERT)

```bash
# Run all neural experiments (Exp1/2/3), 3 seeds by default
python run_neural.py

# Run only one experiment
python run_neural.py --experiment 2

# Run faster sanity pass (LSTM only)
python run_neural.py --experiment 1 --skip-bert

# Optional unified entrypoint
python run_all.py --track all
```

Optional GloVe path (plain text vectors):

```bash
python run_neural.py --glove-path "/path/to/glove.6B.100d.txt"
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

Neural outputs are written under `outputs/neural/`:
- `exp1|exp2|exp3/records.json` — unified per-seed records
- `exp1|exp2|exp3/raw_metrics_by_seed.csv`
- `exp1|exp2|exp3/summary_mean_std.csv`
- `all_records.json`, `all_summary_mean_std.csv`
- `best_bert_hparams.json` (selected from Exp1 val set)

## Reproducibility and fairness constraints

- All model families use deterministic `train/val/test` splits.
- Validation set is used for hyperparameter search and early stopping.
- Test set is evaluated only once per run.
- Exp2 (`IMDB→FPB`) primary metric is `macro-F1`; accuracy is secondary.
- Neural runs use fixed seeds (default: 42, 52, 62) and report `mean ± std`.
- LSTM vocabulary is built from train split only; OOV rates are logged.

## Resource strategy and fallback

- BERT hyperparameter search is performed on Exp1, then best config is reused for Exp2/Exp3.
- If BERT download fails (network/proxy), run `--skip-bert` first to validate the full LSTM pipeline.
- If GloVe file is unavailable, LSTM falls back to random initialization and marks fallback in output config.
