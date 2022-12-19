# Markeing Promotion Classifier
This repsoitory contains all the necessary information to run a classification model. Using Jupyter Notebooks, python scripts, BentoML and docker allow you to access and test this information.  It is also part of my ML-Zoomcamp Capstone Project.

## Data and Inention

The dataset was pulled from Kaggle here: https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign

The target feature in this dataset is the `response` column. This column denotes if a customer responed to the last marketing campaign. We want predict which customers might repsond to a marketing campaign in the future.

Several models were used in this Jupyter Notebook:

- Logistic Regression
- Decision Tree
- Ensemble and Random Forest
- XGBoost


## To run the project

Create a virtual environment on your machine and activate
Clone this repository
Use the requirements.txt to install dependencies
Run the Best Model & BentoML Save notebook for the EDA, model training and parameter tuning.
