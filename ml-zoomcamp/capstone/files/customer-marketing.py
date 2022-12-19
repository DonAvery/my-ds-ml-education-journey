import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

url = "https://raw.githubusercontent.com/DonAvery/my-ds-ml-education-journey/main/ml-zoomcamp/capstone/cleaned-marketing-campaign.csv"
df = pd.read_csv(url)

df_train, df_test = train_test_split(df, test_size=0.2, random_state=11)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = (df_train.response == 'default').astype('int').values
y_test = (df_test.response == 'default').astype('int').values

del df_train['response']
del df_test['response']

dv = DictVectorizer(sparse=False)

train_dicts = df_train.fillna(0).to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

test_dicts = df_test.fillna(0).to_dict(orient='records')
X_test = dv.transform(test_dicts)

rf = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=3, random_state=1)
rf.fit(X_train, y_train)

dtrain = xgb.DMatrix(X_train, label=y_train)

xgb_params = {
    'eta': 0.05,
    'max_depth': 15,
    'min_child_weight': 10,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=130)

import bentoml

bentoml.xgboost.save_model(
    'cust_marketing_model',
    model,
    custom_objects={
        "dictVectorizer": dv
    }
)

