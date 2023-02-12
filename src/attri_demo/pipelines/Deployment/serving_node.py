from mlflow.deployments import get_deploy_client
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.deployments import get_deploy_client
def serve_model(model):
    # alldata = input('Data for prediction:')
    # predictions=model.predict(alldata)
    # alldata["Predictions"]=predictions
    # return pd.DataFrame(alldata)
    return "In development"