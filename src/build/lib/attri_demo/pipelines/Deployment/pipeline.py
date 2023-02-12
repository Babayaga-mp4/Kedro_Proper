from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .model_stuff import *
from .serving_node import serve_model
from .nodes import load_dataset, collect_dataset, feature_engineering, data_viz

def create_pipeline(**kwargs) -> Pipeline:
     return pipeline(
        [
            # node(create_project, inputs=None, outputs="project",name="Create_project") ,
            # node(collect_dataset, inputs=["Data Source 1", "Data Source 2",
            #                            "Data Source 3"], outputs="Raw Data",name="Data_Ingestion"),
            #
            # node(load_dataset, inputs="Raw Data", outputs = ["Pre-Processed Data", "Feature Importance"], name = 'Filtering'),
            #
            # node(data_viz, inputs='Feature Importance', outputs = ["Visualization"]),

            # node(create_dataset,inputs=["processed_dataset","project"],outputs="created_dataset",name="create_dataset"),
            # node(random_string_generator,inputs=None,outputs="UDMI",name="generate_model_name"),
            # node(create_model,inputs=["created_dataset","project","UDMI"],outputs="created_model",name="create_model"),

            # node(feature_engineering, inputs = ['Pre-Processed Data', 'Feature Importance'], outputs = 'Processed Data', name = 'Feature_Engineering'),

            # node(train_test_split_the_data, inputs="Pre-Processed Data",
            #      outputs=["X Train","Y Train","X Test","Y Test"],name="train_test_split"),
            #
            # node(train,inputs=["X Train","Y Train"],outputs="Trained Model",name="Training_The_Model"),
            #
            # node(prediction,inputs=["Trained Model","X Test"],outputs="Predictions",name="predictions_from_the_model"),
            #
            # node(eval_metrics,inputs=["Y Test","Predictions"],outputs="Performance of the Model",name="evaluation_of_the_model"),
            #
            # # node(serve_model,inputs=["ml_model","testx"],outputs="served_predictions",name="model_service"),
            #
            node(mlflow_logging,inputs=["Trained_Model", 'Hyperparameters', "Performance_of_the_Model"],
                 outputs='Logged_Model',name="Logging"),

            node(serve_model,inputs="Logged_Model",outputs=None,name="Deployment")
            ],
         namespace = "Deployment" ,
         inputs = {"Trained_Model", "Hyperparameters", "Performance_of_the_Model"} ,  # map inputs outside of namespace
         outputs = None
     )