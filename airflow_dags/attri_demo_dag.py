from collections import defaultdict

from pathlib import Path

from airflow import DAG
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.version import version
from datetime import datetime, timedelta

from kedro.framework.session import KedroSession
from kedro.framework.project import configure_project


class KedroOperator(BaseOperator):

    @apply_defaults
    def __init__(
        self,
        package_name: str,
        pipeline_name: str,
        node_name: str,
        project_path: str,
        env: str,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.package_name = package_name
        self.pipeline_name = pipeline_name
        self.node_name = node_name
        self.project_path = project_path
        self.env = env

    def execute(self, context):
        configure_project(self.package_name)
        with KedroSession.create(self.package_name,
                                 self.project_path,
                                 env=self.env) as session:
            session.run(self.pipeline_name, node_names=[self.node_name])

# Kedro settings required to run your pipeline
env = "local"
pipeline_name = "__default__"
project_path = Path.cwd()
package_name = "attri_demo"

# Default settings applied to all tasks
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Using a DAG context manager, you don't have to specify the dag property of each task
with DAG(
    "attri-demo",
    start_date=datetime(2019, 1, 1),
    max_active_runs=3,
    schedule_interval=timedelta(minutes=30),  # https://airflow.apache.org/docs/stable/scheduler.html#dag-runs
    default_args=default_args,
    catchup=False # enable if you don't want historical dag runs to run
) as dag:

    tasks = {}

    tasks["data-ingestion-collect-dataset-customer-data-towers-and-complaints-network-logs-cdrs-imei-info-other-data-sources-data-ingestion-raw-data"] = KedroOperator(
        task_id="data-ingestion-collect-dataset-customer-data-towers-and-complaints-network-logs-cdrs-imei-info-other-data-sources-data-ingestion-raw-data",
        package_name=package_name,
        pipeline_name=pipeline_name,
        node_name="Data Ingestion.collect_dataset([Customer_Data,Towers_and_Complaints,Network_Logs,CDRs,IMEI_info,Other_Data_Sources]) -> [Data Ingestion.Raw_Data]",
        project_path=project_path,
        env=env,
    )

    tasks["data-ingestion-filtering"] = KedroOperator(
        task_id="data-ingestion-filtering",
        package_name=package_name,
        pipeline_name=pipeline_name,
        node_name="Data Ingestion.Filtering",
        project_path=project_path,
        env=env,
    )

    tasks["feature-engineering-social-network-analysis"] = KedroOperator(
        task_id="feature-engineering-social-network-analysis",
        package_name=package_name,
        pipeline_name=pipeline_name,
        node_name="Feature Engineering.Social_Network_Analysis",
        project_path=project_path,
        env=env,
    )

    tasks["feature-engineering-statistical-analysis"] = KedroOperator(
        task_id="feature-engineering-statistical-analysis",
        package_name=package_name,
        pipeline_name=pipeline_name,
        node_name="Feature Engineering.Statistical_Analysis",
        project_path=project_path,
        env=env,
    )

    tasks["feature-engineering-feature-selection"] = KedroOperator(
        task_id="feature-engineering-feature-selection",
        package_name=package_name,
        pipeline_name=pipeline_name,
        node_name="Feature Engineering.Feature_Selection",
        project_path=project_path,
        env=env,
    )

    tasks["feature-engineering-feature-transformation"] = KedroOperator(
        task_id="feature-engineering-feature-transformation",
        package_name=package_name,
        pipeline_name=pipeline_name,
        node_name="Feature Engineering.Feature_Transformation",
        project_path=project_path,
        env=env,
    )

    tasks["model-training-train-test-split"] = KedroOperator(
        task_id="model-training-train-test-split",
        package_name=package_name,
        pipeline_name=pipeline_name,
        node_name="Model_Training.train_test_split",
        project_path=project_path,
        env=env,
    )

    tasks["model-training-training-the-model"] = KedroOperator(
        task_id="model-training-training-the-model",
        package_name=package_name,
        pipeline_name=pipeline_name,
        node_name="Model_Training.Training_The_Model",
        project_path=project_path,
        env=env,
    )

    tasks["model-training-predictions-from-the-model"] = KedroOperator(
        task_id="model-training-predictions-from-the-model",
        package_name=package_name,
        pipeline_name=pipeline_name,
        node_name="Model_Training.predictions_from_the_model",
        project_path=project_path,
        env=env,
    )

    tasks["model-training-evaluation-of-the-model"] = KedroOperator(
        task_id="model-training-evaluation-of-the-model",
        package_name=package_name,
        pipeline_name=pipeline_name,
        node_name="Model_Training.evaluation_of_the_model",
        project_path=project_path,
        env=env,
    )

    tasks["deployment-logging"] = KedroOperator(
        task_id="deployment-logging",
        package_name=package_name,
        pipeline_name=pipeline_name,
        node_name="Deployment.Logging",
        project_path=project_path,
        env=env,
    )

    tasks["deployment-deployment"] = KedroOperator(
        task_id="deployment-deployment",
        package_name=package_name,
        pipeline_name=pipeline_name,
        node_name="Deployment.Deployment",
        project_path=project_path,
        env=env,
    )



    tasks["model-training-train-test-split"] >> tasks["model-training-training-the-model"]

    tasks["model-training-train-test-split"] >> tasks["model-training-predictions-from-the-model"]

    tasks["model-training-train-test-split"] >> tasks["model-training-evaluation-of-the-model"]

    tasks["data-ingestion-filtering"] >> tasks["feature-engineering-social-network-analysis"]

    tasks["data-ingestion-filtering"] >> tasks["feature-engineering-statistical-analysis"]

    tasks["model-training-training-the-model"] >> tasks["model-training-predictions-from-the-model"]

    tasks["model-training-training-the-model"] >> tasks["deployment-logging"]

    tasks["model-training-predictions-from-the-model"] >> tasks["model-training-evaluation-of-the-model"]

    tasks["feature-engineering-statistical-analysis"] >> tasks["feature-engineering-feature-selection"]

    tasks["feature-engineering-statistical-analysis"] >> tasks["feature-engineering-feature-transformation"]

    tasks["feature-engineering-social-network-analysis"] >> tasks["feature-engineering-feature-selection"]

    tasks["feature-engineering-social-network-analysis"] >> tasks["feature-engineering-feature-transformation"]

    tasks["data-ingestion-collect-dataset-customer-data-towers-and-complaints-network-logs-cdrs-imei-info-other-data-sources-data-ingestion-raw-data"] >> tasks["data-ingestion-filtering"]

    tasks["feature-engineering-feature-transformation"] >> tasks["model-training-train-test-split"]

    tasks["feature-engineering-feature-selection"] >> tasks["feature-engineering-feature-transformation"]

    tasks["model-training-evaluation-of-the-model"] >> tasks["deployment-logging"]

    tasks["deployment-logging"] >> tasks["deployment-deployment"]
