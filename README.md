# NLP Final Project — Domain Shift in Sentiment Classification

**Research Question:** How do different NLP models vary in their ability to generalize from general-domain sentiment data to financial-domain sentiment classification under domain shift?

## Project Structure

```text
nlp_project/
├── data_loader.py          # Dataset loading, cleaning, and train/val/test splits
├── features.py             # TF-IDF feature extraction for classical models
├── models.py               # Classical model training (NB / LR)
├── evaluate.py             # Metrics, confusion matrices, summary tables, plots
├── result_schema.py        # Unified experiment record schema
├── run_classical.py        # Classical experiment runner
├── neural_features.py      # Tokenization, vocabulary, GloVe loading, dataloaders
├── neural_models.py        # LSTM and BERT training / evaluation code
├── run_neural.py           # Neural experiment runner
├── run_and_log_neural.py   # Neural runner with command logging
├── run_all.py              # Unified entrypoint for classical / neural tracks
├── requirements.txt
├── README.md
└── outputs/                # Generated metrics, plots, and summaries
```

## Datasets

We use two public datasets:

- **IMDB Reviews**: 50,000 labeled movie reviews for binary sentiment classification.
- **Financial PhraseBank (FPB)**: financial news / statement sentences with positive, neutral, and negative labels.

To align FPB with binary IMDB sentiment, **neutral FPB examples are removed** and the remaining labels are remapped to binary sentiment.

## Experiments

We run three core experiments:

1. **Experiment 1 — In-domain general sentiment (`IMDB → IMDB`)**  
   Train, validate, and test on IMDB.
2. **Experiment 2 — Cross-domain transfer (`IMDB → FPB`)**  
   Train on IMDB, validate on IMDB, test on Financial PhraseBank.
3. **Experiment 3 — In-domain financial sentiment (`FPB → FPB`)**  
   Train, validate, and test on Financial PhraseBank.

## Models

### Classical baselines

- **Multinomial Naive Bayes** with TF-IDF features
- **Complement Naive Bayes** with TF-IDF features
- **Logistic Regression** with TF-IDF features
  - C in `{0.1, 1.0, 10.0}`
  - with / without balanced class weights

### Neural models

- **LSTM + pretrained GloVe embeddings**
- **BERT (`bert-base-uncased`) fine-tuning**

## Evaluation

Reported metrics include:

- Accuracy
- Precision
- Recall
- F1
- Macro-F1

We also generate:

- Confusion matrices
- Per-experiment comparison tables
- Cross-domain performance-drop summaries
- Fine-tuning improvement summaries

## Reproducibility and fairness constraints

- All model families use deterministic `train/val/test` splits.
- Validation is used for hyperparameter search and early stopping.
- Test data is used only for final evaluation.
- For **Exp2 (`IMDB → FPB`)**, **macro-F1** is the primary metric and accuracy is secondary.
- Neural runs use fixed seeds (default: `42 52 62`) and report **mean ± std**.
- LSTM vocabulary is built from the training split only.
- If GloVe is unavailable, the LSTM falls back to random initialization and records the fallback in output config.

## Setup

Create / activate your Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Classical experiments

Run all classical experiments:

```bash
python run_classical.py
```

Run one experiment only:

```bash
python run_classical.py --experiment 1   # IMDB -> IMDB
python run_classical.py --experiment 2   # IMDB -> FPB
python run_classical.py --experiment 3   # FPB -> FPB
```

## Neural experiments

### Important note about GloVe

`glove.6B.100d.txt` is a large external file and **should not be committed to GitHub**.
Download it locally, keep it outside version control, and pass its path with `--glove-path`.

Typical local usage from the project root:

```bash
python run_neural.py --experiment 1 --seeds 42 --glove-path "glove.6B.100d.txt" --skip-bert
```

### Recommended execution order

#### 1. Smoke test: LSTM only, one seed

```bash
python run_neural.py --experiment 1 --seeds 42 --glove-path "glove.6B.100d.txt" --skip-bert
```

#### 2. Smoke test: full neural pipeline, one seed

```bash
python run_neural.py --experiment 1 --seeds 42 --glove-path "glove.6B.100d.txt"
python run_neural.py --experiment 2 --seeds 42 --glove-path "glove.6B.100d.txt"
python run_neural.py --experiment 3 --seeds 42 --glove-path "glove.6B.100d.txt"
```

#### 3. Final neural runs: three seeds

```bash
python run_neural.py --experiment 1 --seeds 42 52 62 --glove-path "glove.6B.100d.txt"
python run_neural.py --experiment 2 --seeds 42 52 62 --glove-path "glove.6B.100d.txt"
python run_neural.py --experiment 3 --seeds 42 52 62 --glove-path "glove.6B.100d.txt"
```

### Other useful neural commands

Run all neural experiments:

```bash
python run_neural.py --glove-path "glove.6B.100d.txt"
```

Run LSTM-only for all experiments:

```bash
python run_neural.py --seeds 42 52 62 --glove-path "glove.6B.100d.txt" --skip-bert
```

Run with logging:

```bash
python run_and_log_neural.py --experiment 1 --seeds 42 --glove-path "glove.6B.100d.txt"
```

Use the unified entrypoint:

```bash
python run_all.py --track all
```

## Resource strategy

- BERT hyperparameter search is performed on **Exp1** first.
- The selected BERT setting is then reused for **Exp2** and **Exp3**.
- For quick debugging, start with `--seeds 42` before running all three seeds.
- If BERT download or initialization fails, use `--skip-bert` to validate the full LSTM pipeline first.

## Outputs

### Classical outputs (`outputs/`)

- `exp{1,2,3}_comparison.csv` — metrics tables by experiment
- `exp{1,2,3}_cm_*.png` — confusion matrices
- `exp{1,2,3}_results.json` — raw metrics
- `summary_all_experiments.csv` — combined summary
- `summary_performance_drop.png` — in-domain vs cross-domain comparison
- `summary_finetune_improvement.png` — cross-domain vs fine-tuned comparison

### Neural outputs (`outputs/neural/`)

- `exp1/`, `exp2/`, `exp3/`
- `records.json` — unified per-seed records
- `raw_metrics_by_seed.csv`
- `summary_mean_std.csv`
- `all_records.json`
- `all_summary_mean_std.csv`
- `best_bert_hparams.json`

## Git / repository notes

Recommended `.gitignore` entries:

```gitignore
__pycache__/
*.pyc
outputs/
glove.6B.100d.txt
```

Do **not** upload large pretrained files such as `glove.6B.100d.txt` to GitHub. Keep them local and document the download / path instead.
