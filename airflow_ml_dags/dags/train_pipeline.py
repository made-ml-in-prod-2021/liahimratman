from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor

from utils import default_args, VOLUME


with DAG(
        "train_pipeline",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(5),
) as dag:
    start = DummyOperator(task_id="Begin")

    data_sensor = FileSensor(
        task_id="Wait_for_data",
        poke_interval=10,
        retries=100,
        filepath="data/raw/{{ ds }}/data.csv"
    )

    target_sensor = FileSensor(
        task_id="Wait_for_target",
        poke_interval=10,
        retries=100,
        filepath="data/raw/{{ ds }}/target.csv"
    )

    preprocess = DockerOperator(
        task_id="Data_preprocess",
        image="airflow-preprocess",
        command="/data/raw/{{ ds }} /data/processed/{{ ds }} /data/model/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[VOLUME],
    )

    split = DockerOperator(
        task_id="Split_data",
        image="airflow-split",
        command="/data/processed/{{ ds }} /data/splitted/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    train = DockerOperator(
        task_id="Train_model",
        image="airflow-train",
        command="/data/splitted/{{ ds }} /data/model/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    validate = DockerOperator(
        task_id="Validate_model",
        image="airflow-validate",
        command="/data/splitted/{{ ds }} /data/model/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[VOLUME]
    )

    finish = DummyOperator(task_id="End")

    start >> [data_sensor, target_sensor] >> preprocess >> split >> train >> validate >> finish
