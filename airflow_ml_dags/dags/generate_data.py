from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from utils import default_args, VOLUME


with DAG(
        "generate_data",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:
    start = DummyOperator(task_id="Begin")

    download = DockerOperator(
        task_id="Generate_data",
        image="airflow-generate",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[VOLUME],
    )

    finish = DummyOperator(task_id="End")

    start >> download >> finish
