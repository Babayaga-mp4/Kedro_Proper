from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
     return pipeline(
        [

            node(statistical_feature_engineering, inputs=["Pre-Processed Data"],
                 outputs='Statistical_Features',name="Statistical_Feature_Engineering"),

            node(SNA_feature_engineering ,inputs = ["Pre-Processed Data"] ,
                 outputs = 'SNA_Features' ,name = "SNA_Feature_Engineering") ,

            node(feature_selection ,inputs = ["SNA_Features", "Statistical_Features"] ,
                 outputs = 'selected_features' ,name = "Feature_Selection") ,

            node(feature_transformation,inputs = ["SNA_Features", "Statistical_Features", 'selected_features'],
                 outputs = 'dataset' ,name = "Feature_Transformation") ,
            ],
         namespace = "Feature Engineering",
         inputs = {"Pre-Processed Data"} ,  # map inputs outside of namespace
         outputs = "dataset"
     )