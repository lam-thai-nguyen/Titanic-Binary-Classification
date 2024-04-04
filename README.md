# Logistic Regression for famous Kaggle Titanic dataset

Dataset source: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

## Features and Target

- Target: Survived (0 or 1)
- Features:
  - Pclass 
  - Sex
  - Age
  - SibSp
  - Parch
  - Fare
  - Embarked

## Recommended viewing steps

- [README](README.md) -> [EDA](EDA.ipynb) -> [Logistic Regression](main.ipynb) --Optional--> [LR utils](utils.py), [LR theory](example.ipynb)

## Script description

- [titanic_dataset](titanic_dataset) folder: contains 3 csv file
  - train.csv -> train the model
  - test.csv -> for prediction
  - gender_submission.csv -> final prediction file needs to follow the format of this file
- [EDA](EDA.ipynb): Exploratory Data Analysis
- [example](example.ipynb): LR theory
- [main](main.ipynb): LR implementation
- [utils](utils.py): LR from scratch
- [submission](submission.csv): final prediction file
- [requirements](requirements.txt): used libraries 

## Expectation

- Better understanding how Logistic Regression works
- Implementing Logistic Regression from scratch as well as using scikit-learn