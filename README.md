# Movie Recommendation Metaflow Pipeline

This project packages a lightweight movie-title classification workflow as a reproducible Metaflow pipeline. It takes local movie metadata, engineers a simple popularity signal, trains a logistic regression baseline, and exports evaluation artifacts that make the run easy to inspect.

## Overview

The workflow is built around a small end-to-end ML pipeline:

- load movie metadata from a local CSV source
- extract release-year information from titles
- create a synthetic binary popularity label for experimentation
- vectorize titles with bag-of-words features
- train and evaluate a logistic regression classifier
- generate a confusion matrix image and classification report

## Tech Stack

- Python
- Metaflow
- pandas
- scikit-learn
- matplotlib

## Project Structure

```text
.
├── data/
│   └── movies.csv
├── movie_popularity_flow.py
├── classification_report.csv
├── confusion_matrix.png
└── requirements.txt
```

## How It Works

The pipeline is defined in [movie_popularity_flow.py](/Users/srivatsakamballa/Desktop/My-Projects/Movie-Recommendation_Metaflow/movie_popularity_flow.py:1) and runs through these stages:

1. `start`: load the movie dataset from `data/movies.csv`
2. `clean_data`: drop incomplete rows, extract release year, and build an experimental popularity label
3. `vectorize_data`: convert titles into bag-of-words features with `CountVectorizer`
4. `train_model`: split data, train logistic regression, and compute evaluation metrics
5. `report_results`: save a confusion matrix plot and classification report
6. `end`: complete the run

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python movie_popularity_flow.py run
```

## Outputs

- `classification_report.csv`: precision, recall, f1-score, and accuracy summary
- `confusion_matrix.png`: saved evaluation plot from the latest run
- Metaflow card output: visual summary for the pipeline execution

## Notes

- The popularity target is synthetic and intended for workflow experimentation, not production recommendation quality.
- This repo is best positioned as a compact MLOps / workflow orchestration project rather than a full recommendation engine.
