# ShopSmart Retail Ltd. — Machine Learning Project

## Project Overview

This project applies machine learning techniques to retail data from ShopSmart Retail Ltd. The goal is to explore how machine learning can support order-value prediction and return-risk analysis.

The project covers data preparation, exploratory analysis, model development, evaluation, model persistence and prediction using saved models.

## Objectives

- Prepare and combine retail datasets for machine learning.
- Develop a regression model to predict order value.
- Develop a classification model to predict whether an order will be returned.
- Evaluate model performance using appropriate metrics.
- Save trained models and demonstrate predictions using Python.

## Machine Learning Models

### 1. Linear Regression

Linear Regression was used to predict `Order_Total`, a numerical target.

The baseline model achieved the following results:

| Metric | Result |
|---|---:|
| MAE | 29,538.14 |
| RMSE | 40,079.22 |
| R² | 0.8875 |

Feature engineering also revealed that the target could be reconstructed from variables used in its calculation. This produced near-perfect results but represents **target leakage**, rather than evidence of genuine predictive performance. The result should therefore be interpreted cautiously.

### 2. Logistic Regression

Logistic Regression was used to predict `Return_Flag`, a binary target indicating whether an order was returned.

Because the returned-order class was underrepresented, a class-weighted model was also evaluated.

| Metric for returned orders | Result |
|---|---:|
| Precision | 0.0463 |
| Recall | 0.4529 |
| F1-score | 0.0844 |

The model identified some returned orders, but its low precision means that many orders flagged as potential returns were not actually returned. Further improvement is needed before the model could support operational decisions.

## Data Preparation

The project combined nine source datasets and created a master dataset containing 50,000 records and 37 columns.

The processed data includes:

- `master_dataset.csv` — the combined dataset.
- `train.csv` — the training dataset containing 40,000 records.
- `test.csv` — the test dataset containing 10,000 records.

The source datasets are kept locally in `raw_data/` and are excluded from the Git repository.

## Project Structure

```text
Week1 Machine Learning/
├── ML01_Foundations.ipynb
├── requirements.txt
├── .gitignore
├── data/
│   ├── master_dataset.csv
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── linear_regression.pkl
│   ├── logistic_regression.pkl
│   ├── encoder.pkl
│   └── scaler.pkl
├── scripts/
│   └── predict.py
├── reports/
│   ├── ML_Report.pdf
│   └── Presentation.pdf
└── raw_data/  (local only; excluded from Git)
```

## Setup and Installation

Clone or download the repository, then open a terminal in the project directory.

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
pip install -r requirements.txt
```

## Run the Prediction Script

From the project root, run:

```powershell
python scripts/predict.py
```

The script loads the saved models and preprocessing objects from `models/`, accepts example input values and displays an estimated order value and return probability.

## Limitations

- The regression target was constructed from order-related values, creating a risk of target leakage when those same components are used as features.
- The return classification model has low precision for returned orders.
- Model performance should be improved and validated before using predictions to make real business decisions.
- The findings depend on the available datasets and the assumptions made during data preparation.

## Conclusion

This project demonstrates an end-to-end machine learning workflow, from preparing retail data to training, evaluating, saving and using predictive models. The results highlight both the potential of machine learning for retail analysis and the importance of checking for target leakage and evaluating model performance carefully.