# Predicting S&P 500 Direction with Machine Learning

Scientific research project (SGU 2026): can classical ML models predict whether the S&P 500 closes higher tomorrow, and does a long-only strategy built on those predictions beat buy-and-hold?

## Pipeline

| # | Notebook | Purpose | Output |
|---|----------|---------|--------|
| 01 | [`collectdata_s&p_500.ipynb`](collectdata_s%26p_500.ipynb) | Download `^GSPC` (2010–2025) from Yahoo Finance and build the next-day target | `sp500_supervised_data.csv` |
| 02 | [`raw_eda.ipynb`](raw_eda.ipynb) | Data-quality checks, class balance, price history | figures |
| 03 | [`data_processing.ipynb`](data_processing.ipynb) | Walk-forward (3-year train / 1-year test) Random Forest with per-window scaling | out-of-sample predictions |
| 04 | [`Experiment_Machine_Learning.ipynb`](Experiment_Machine_Learning.ipynb) | Compare Logistic Regression, SVM, Random Forest, XGBoost; tune with `RandomizedSearchCV`; back-test vs. Buy & Hold | metrics, equity curves |

Each notebook has an *Open in Colab* badge at the top.

## Methodology

- **Target:** `1` if tomorrow's close > today's close, else `0` (≈ 54.7 % up days).
- **Evaluation:** Accuracy, Precision, Recall, F1, confusion matrix, cumulative return and maximum drawdown on a chronological hold-out set (765 days).
- **Strategy:** hold the index on predicted *up* days, stay in cash otherwise.

## Key findings (hold-out set)

| Model | Accuracy | F1 | Max drawdown |
|-------|---------:|---:|-------------:|
| Logistic Regression | 53.5 % | 0.645 | −23.7 % |
| SVM | 50.1 % | 0.542 | −20.8 % |
| Random Forest | 46.5 % | 0.306 | −17.2 % |
| XGBoost | 46.3 % | 0.309 | −18.5 % |

No model clearly beats the ~56 % "always up" baseline, which is expected for a near-efficient market. The tuned SVM collapses to predicting *up* every day.

## Reproducing

```bash
pip install yfinance pandas numpy scikit-learn xgboost matplotlib seaborn
```

Run the notebooks in order (Colab or Jupyter).

## Known gaps

- The feature-engineering step that creates `sp500_supervised_features_final.csv`, `sp500_train_80.csv` and `sp500_test_20.csv` is not in this repository yet.
- `RandomizedSearchCV` uses shuffled K-fold; `TimeSeriesSplit` would avoid look-ahead leakage in the tuning step.
- Saved notebook outputs for the charts predate the restyle; re-run to refresh them.
