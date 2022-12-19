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

Run the customer-marketing notebook for the EDA, model training and parameter tuning.

When ready for deployment run the Best Model & BentoML Save notebook.

After running the BentoML notebook you can start setting up the deployment.

Run bentoml build inside the folder with the files from the files directory.

Run bentoml containerize <tag> the tag will be given to you after the build.

Run docker run -it --rm -p 3000:3000 <tag> serve --production, this line will be given after docker run, copy and paste it.

Copy the json from the locustfile.py and use http://0.0.0.0:8089 to post that json file into the get portion of the page and see the results.

