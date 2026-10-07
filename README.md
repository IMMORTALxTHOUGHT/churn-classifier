# Bank Customer Churn Classifier

Predict whether a bank customer will churn using Logistic Regression and Random Forest, with an honest evaluation of model limitations.

## Project Structure

```
├── app.py                          # Streamlit demo app
├── notebooks/
│   └── 00-full-pipeline.ipynb      # EDA → training → error analysis
├── data/
│   └── Churn_Modelling.csv         # Kaggle dataset (10,000 customers)
├── outputs/                        # Trained model files (gitignored)
├── presentation.pdf                # 2-page summary
├── requirements.txt
└── .gitignore
```

## Results (Test Set)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression | 80.8% | 60.2% | 17.4% | 27.0% |
| Random Forest (tuned) | ~80% | ~45% | ~77% | ~57% |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Run the Notebook

```bash
jupyter-notebook notebooks/00-full-pipeline.ipynb
```

The notebook includes EDA, baseline model, improved model with threshold tuning, and 3 real error cases (FP, FN, TP).

## Dataset

Bank Customer Churn from Kaggle — 10,000 rows, ~20.4% churn rate. Features: credit score, geography, gender, age, tenure, balance, products, credit card status, active membership, estimated salary.

## Key Takeaway

Accuracy alone is misleading (79.6% by predicting all "stayed"). The models reveal a precision-recall trade-off: LR is well-calibrated but misses most churners; RF catches more at the cost of inflated probabilities.
# CAT tick 2026-09-28_12:30:28 tick=1790598628
# CAT tick 2026-09-29_09:17:01 tick=1790673421
# CAT tick 2026-09-29_09:30:35 tick=1790674235
# CAT tick 2026-09-29_12:30:34 tick=1790685034
# CAT tick 2026-09-30_06:30:19 tick=1790749819
# CAT tick 2026-09-30_10:40:59 tick=1790764859
# CAT tick 2026-09-30_14:45:48 tick=1790779548
# CAT tick 2026-10-02_15:41:38 tick=1790955698
# CAT tick 2026-10-03_12:14:29 tick=1791029669
# CAT tick 2026-10-03_12:14:31 tick=1791029671
# CAT tick 2026-10-03_12:14:31 tick=1791029671
# CAT tick 2026-10-03_12:30:21 tick=1791030621
# CAT tick 2026-10-03_12:30:26 tick=1791030626

# CAT tick 2026-10-04_09:24:44

# CAT tick 2026-10-06_11:08:48

# CAT tick 2026-10-07_14:21:21

# CAT tick 2026-10-07_14:21:33
