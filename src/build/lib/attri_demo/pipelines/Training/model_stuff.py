# Gets mlflow up and running.
# mlflow server - -backend - store - uri sqlite: ///db/mlflow.db - -default - artifact - root. / mlruns

import logging
from urllib.parse import urlparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier ,AdaBoostClassifier
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
import time
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
import xgboost as xgb
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score, precision_score, balanced_accuracy_score, recall_score,
                             log_loss, zero_one_loss)
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger()


def train_test_split_the_data(data):
    data.churn.replace({"yes": 1 ,"no": 0} ,inplace = True)
    dummy_df = pd.get_dummies(data)
    y = dummy_df.churn.values
    X = dummy_df.drop('churn' ,axis = 1)
    cols = X.columns

    smt = SMOTE(sampling_strategy = 0.7)
    X ,y = smt.fit_resample(X ,y)

    X_train ,X_test ,y_train ,y_test = train_test_split(X ,y ,test_size = .25 ,random_state = 33)
    mm = MinMaxScaler().fit(X_train)

    X_train = pd.DataFrame(mm.transform(X_train))
    X_train.columns = cols
    X_test = pd.DataFrame(mm.transform(X_test))
    X_test.columns = cols

    return X_train, y_train, X_test, y_test

class_weights = {0:1, 1:15}

def train(X_train, y_train, Hyperparameters):
    mlflow.sklearn.autolog()
    lr = LogisticRegression(class_weight = class_weights ,random_state = 42)
    lr.fit(X_train ,y_train)
    create_engine("sqlite:///mlflow.db" ,pool_pre_ping = True)
    mlflow.set_registry_uri(
        "sqlite:///mlflow.db")
    # mlflow.set_tracking_uri('sqlite:///mlflow.db') # Set in mlflow.yml
    confusion_matrix ,precision_recall_curve ,roc_curve = [1,1,1] # dummy
    return lr,confusion_matrix, precision_recall_curve, roc_curve


def eval_metrics(actual, raw_preds) -> pd.DataFrame:
    pred = raw_preds.to_numpy()
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    accuracy = accuracy_score(actual, pred)
    report = precision_score(actual, pred)
    balanced_acc = balanced_accuracy_score(actual, pred)
    recall = recall_score(actual, pred)
    lloss = log_loss(actual, pred)
    zoloss = zero_one_loss(actual, pred)


    df = {}
    df["rmse"] = rmse
    df["mae"] = mae
    df["r2"] = r2
    df["accuracy_value"] = accuracy
    df['precision_score'] = report
    df['balanced_acc'] = balanced_acc
    df['recall'] = recall
    df['lloss'] = lloss
    df['zoloss'] = zoloss

    return pd.DataFrame(df,index=[0])

def prediction(lr, X_test):
    predicted_qualities = lr.predict(X_test)
    return pd.DataFrame(predicted_qualities ,columns = ['Predictions'])

def mlflow_logging(model, data):
    with mlflow.start_run(run_name = "Logistic Regression" ,nested = True):
            mlflow.log_param('class_weights', class_weights)
            mlflow.log_metric("accuracy_score" ,data.iloc[0]['accuracy_value'])
            mlflow.log_metric("precision" ,data.iloc[0]['precision_score'])
            mlflow.log_metric("rmse" ,data.iloc[0]['rmse'])
            mlflow.log_metric("log_loss" ,data.iloc[0]['lloss'])
            mlflow.log_metric("r2" ,data.iloc[0]['r2'])
            mlflow.log_metric("zero-one_loss" ,data.iloc[0]['zoloss'])
            mlflow.log_metric("mae" ,data.iloc[0]['mae'])
            mlflow.log_metric("balanced_accuracy_score" ,data.iloc[0]['balanced_acc'])
            mlflow.log_metric("recall_score" ,data.iloc[0]['recall'])

            mlflow.sklearn.log_model(
                sk_model = model,
                artifact_path = "sklearn-model")

            model_uri = "runs:/{}/sklearn-model".format(mlflow.active_run().info.run_id)
            mlflow.register_model(model_uri ,"Churn Prediction")
    return model

          
