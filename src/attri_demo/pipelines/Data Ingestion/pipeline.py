from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import load_dataset, collect_dataset, data_viz

def create_pipeline(**kwargs) -> Pipeline:
     return pipeline(
        [
            node(collect_dataset, inputs=["Customer_Data", "Towers_and_Complaints",
                                       "Network_Logs", "CDRs", "IMEI_info", "Other_Data_Sources"], outputs="Raw_Data"),

            node(load_dataset, inputs="Raw_Data", outputs = "Pre-Processed Data", name = 'Filtering'),


            # node(data_viz, inputs='Feature Importance', outputs = ["Visualization"]),

            ],
         namespace = "Data Ingestion" ,
         inputs = {"Customer_Data", "Towers_and_Complaints",
                                       "Network_Logs", "CDRs", "IMEI_info", "Other_Data_Sources"} ,
         outputs = {
             "Pre-Processed Data" ,
             # "Visualization" ,
         }
     )