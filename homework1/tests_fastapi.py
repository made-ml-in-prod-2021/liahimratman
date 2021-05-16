from typing import List

from faker import Faker
from fastapi.testclient import TestClient
import pandas as pd
from pydantic import parse_obj_as
import pytest

from api import app
from online_inference.schemas import DiseaseModel, DiseaseResponseModel


@pytest.fixture(scope="session")
def fake_dataset_path():
    return "tests_data/fake_dataset_test.csv"


@pytest.fixture(scope="session")
def categorical_features() -> List[str]:
    return [
        "cp",
        "sex",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture(scope="session")
def test_client():
    client = TestClient(app)
    return client


def make_fake_dataset(path):
    """
    Make and save fake dataset
    :param path: saving path
    :return: None
    """
    fake = Faker()
    fake.set_arguments('age', {'min_value': 1, 'max_value': 100})
    fake.set_arguments('trestbps', {'min_value': 80, 'max_value': 200})
    fake.set_arguments('chol', {'min_value': 120, 'max_value': 600})
    fake.set_arguments('restecg_slope', {'min_value': 0, 'max_value': 2})
    fake.set_arguments('thalach', {'min_value': 70, 'max_value': 210})
    fake.set_arguments('oldpeak', {'min_value': 0, 'max_value': 7})
    fake.set_arguments('ca', {'min_value': 0, 'max_value': 4})
    fake.set_arguments('thal_cp', {'min_value': 0, 'max_value': 3})
    fake.set_arguments('binary', {'min_value': 0, 'max_value': 1})
    fake_data = fake.csv(
        header=("age", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
                "oldpeak", "slope", "ca", "thal", "cp", "sex", "target"),
        data_columns=('{{pyint:age}}', '{{pyint:trestbps}}', '{{pyint:chol}}',
                      '{{pyint:binary}}', '{{pyint:restecg_slope}}',
                      '{{pyint:thalach}}', '{{pyint:binary}}', '{{pyfloat:oldpeak}}',
                      '{{pyint:restecg_slope}}', '{{pyint:ca}}', '{{pyint:thal_cp}}',
                      '{{pyint:thal_cp}}', '{{pyint:binary}}', '{{pyint:binary}}'),
        num_rows=100,
        include_row_ids=False).replace('\r', '')

    with open(path, 'w') as input_stream:
        input_stream.write(fake_data)


@pytest.fixture(scope="session")
def dataset(fake_dataset_path: str) -> pd.DataFrame:
    make_fake_dataset(fake_dataset_path)
    data = pd.read_csv(fake_dataset_path)
    data.drop(columns=["target"], inplace=True)

    return data


def test_predict(dataset: pd.DataFrame, test_client: TestClient):
    ids = list(range(dataset.shape[0]))
    request = DiseaseModel(
        ids=ids,
        features=dataset.values.tolist(),
        columns=dataset.columns.tolist()
    )
    with test_client as client:
        response = client.post("/predict", data=request.json())
    preds = parse_obj_as(List[DiseaseResponseModel], response.json())
    assert response.status_code == 200
    assert len(preds) == dataset.shape[0]
    assert set([i.id for i in preds]) == set(ids)


def test_predict_wrong_shape(dataset: pd.DataFrame, test_client: TestClient):
    ids = list(range(dataset.shape[0]))
    request = DiseaseModel(
        ids=ids,
        features=dataset.values.tolist(),
        columns=dataset.columns.tolist()[:-1]
    )
    with test_client as client:
        response = client.post("/predict", data=request.json())
    assert response.status_code == 400


def test_predict_wrong_column(dataset: pd.DataFrame, test_client: TestClient):
    ids = list(range(dataset.shape[0]))
    columns = dataset.columns.tolist()[:-1] + ["extra_column"]
    request = DiseaseModel(
        ids=ids,
        features=dataset.values.tolist(),
        columns=columns
    )
    with test_client as client:
        response = client.post("/predict", data=request.json())
    assert response.status_code == 400


def test_predict_wrong_dtype(dataset: pd.DataFrame, test_client: TestClient, categorical_features: List[str]):
    dataset_copy = dataset.copy(deep=True)
    ids = list(range(dataset_copy.shape[0]))
    dataset_copy[categorical_features[0]] = float('nan')
    request = DiseaseModel(
        ids=ids,
        features=dataset_copy.values.tolist(),
        columns=dataset_copy.columns.tolist()
    )
    with test_client as client:
        response = client.post("/predict", data=request.json())
    assert response.status_code == 400
