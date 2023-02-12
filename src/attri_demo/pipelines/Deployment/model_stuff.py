# Gets mlflow up and running.
# mlflow server - -backend - store - uri sqlite: ///db/mlflow.db - -default - artifact - root. / mlruns

import logging
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn



logging.basicConfig(level=logging.WARN)
logger = logging.getLogger()

class_weights = {0:1, 1:15}


def mlflow_logging(model, Hyperparameters, data):
    # with mlflow.start_run(run_name = "Logistic Regression" ,nested = False):
            mlflow.sklearn.autolog()
            mlflow.set_tag("mlflow.runName" ,"Logistic Regression")
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

            # mlflow.sklearn.log_model(
            #     sk_model = model,
            #     artifact_path = "sklearn-model")

            model_uri = "runs:/{}/sklearn-model".format(mlflow.active_run().info.run_id)
            mlflow.register_model(model_uri ,"Logistic Regression")
            return model


          
