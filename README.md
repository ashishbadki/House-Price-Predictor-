# House Price Predictor

Predicting residential property prices in Mumbai, India using machine learning.

> **Status:** Phase 1 — project setup. This README is a placeholder.
> It gets rewritten properly in Phase 25.

## Project structure

```
house-price-predictor/
├── data/
│   ├── raw/          # original, never-edited dataset
│   └── processed/    # cleaned output of our pipeline
├── notebooks/        # exploration and experiments
├── src/              # reusable, tested Python modules
├── models/           # saved model pipeline (.joblib)
├── app/              # Streamlit web application
├── tests/            # automated tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
conda create -n houseprice python=3.11 -y
conda activate houseprice
pip install -r requirements.txt
```

## Disclaimer

Predictions produced by this project are statistical estimates based on
listing data. Listing prices are **asking prices**, not confirmed transaction
prices, so an estimate may differ substantially from an actual sale price.
