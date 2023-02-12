from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .model_stuff import *
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
     return pipeline(
        [
            node(train_test_split_the_data, inputs="dataset",
                 outputs=["X_Train","y_Train","X_Test","y_Test"],name="train_test_split"),


            node(train,inputs=["X_Train","y_Train", "Hyperparameters"],outputs=["Trained_Model", "confusion_matrix", "precision_recall_curve", "roc_curve"],name="Training_The_Model"),

            node(prediction,inputs=["Trained_Model","X_Test"],outputs="Predictions",name="predictions_from_the_model"),

            node(eval_metrics,inputs=["y_Test","Predictions"],outputs="Performance_of_the_Model",name="evaluation_of_the_model"),

            ],
     namespace = "Model_Training",
                 inputs = {"dataset", "Hyperparameters"} ,
                          outputs = {
                            "Performance_of_the_Model",
                            "Trained_Model",
                              "confusion_matrix", "precision_recall_curve", "roc_curve"
                          }
     )