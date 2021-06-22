from datetime import timedelta

VOLUME = 'C:/Users/Mikhail Korotkov/PycharmProjects/untitled/liahimratman/airflow_ml_dags/data/:/data'

default_args = {
    "owner": "Korotkov_Mikhail",
    "email": ["ma-korotkov-ml@yandex.ru"],
    "email_on_failure": True,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}
