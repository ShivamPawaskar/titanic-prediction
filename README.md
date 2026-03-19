# Titanic Survival Prediction

Machine learning project for the Kaggle Titanic competition using feature engineering, hyperparameter tuning, and ensemble models built in Jupyter.

## Project Overview

This notebook predicts passenger survival on the Titanic dataset. The workflow includes:

- loading the Kaggle `train.csv` and `test.csv` files
- feature engineering from names, tickets, cabins, and family information
- leave-one-out group survival features to capture family and ticket-level patterns
- Optuna tuning for XGBoost and LightGBM
- comparison of multiple classifiers with 10-fold stratified cross-validation
- final prediction through stacking or soft-voting ensemble selection

## Files

- `Titanic_Prediction.ipynb` - main notebook with data preparation, training, evaluation, and submission generation
- `train.csv` - training dataset
- `test.csv` - test dataset
- `submission.csv` - generated submission file
- `gender_submission.csv` - Kaggle reference submission file

## Models Used

- XGBoost
- LightGBM
- Random Forest
- Extra Trees
- HistGradientBoosting
- Support Vector Classifier
- Logistic Regression meta-learner for stacking

## Feature Engineering Highlights

- extracted passenger titles from names
- grouped rare titles
- built family size and family category features
- extracted cabin deck information
- filled missing values for age, fare, and embarkation
- created leave-one-out survival features using ticket groups and last names
- label-encoded categorical variables

## Tech Stack

- Python
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- optuna
- matplotlib
- seaborn

## How To Run

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook Titanic_Prediction.ipynb
```

4. Run the cells in order to train models and regenerate `submission.csv`.

## Notes

- The notebook installs some packages inline as well, but `requirements.txt` is included for cleaner setup.
- Model selection is based on cross-validation performance before generating the final submission.
- `submission.csv` contains predictions for the Kaggle test set.

## Dataset Source

Dataset files are from the Kaggle Titanic competition:
https://www.kaggle.com/competitions/titanic

## Author

Shivam Pawaskar
