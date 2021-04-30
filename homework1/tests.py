import os
import pytest
import pickle
from typing import List, Tuple
import pandas as pd
from py._path.local import LocalPath
from sklearn.linear_model import LogisticRegression

from ml_project.data_functions.make_dataset import read_data, split_train_val_data
from ml_project.params.split_params import SplittingParams
from ml_project.params.train_params import TrainingParams
from ml_project.params.feature_params import FeatureParams
from ml_project.features.build_features import make_features, extract_target, build_transformers
from ml_project.models.model_fit_predict import train_model, serialize_model


@pytest.fixture()
def dataset_path():
    return "tests_data/train_data_sample.csv"


@pytest.fixture()
def target_col():
    return "target"


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "cp",
        "sex",
    ]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]


def test_load_dataset(dataset_path: str, target_col: str):
    data = read_data(dataset_path)
    assert len(data) > 10
    assert target_col in data.keys()


def test_split_dataset(dataset_path: str):
    val_size = 0.2
    splitting_params = SplittingParams(random_state=239, val_size=val_size,)
    data = read_data(dataset_path)
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] > 10
    assert val.shape[0] > 10


@pytest.fixture
def features_and_target(
    dataset_path: str, categorical_features: List[str], numerical_features: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col="target",
    )
    data = read_data(dataset_path)
    transformers = build_transformers(params)
    features, _ = make_features(transformers, data, mode="train")
    target = extract_target(data, params)
    return features, target


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams(model_type="LogisticRegression"))
    assert isinstance(model, LogisticRegression)
    assert model.predict(features).shape[0] == target.shape[0]


def test_serialize_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    model = LogisticRegression(max_iter=1000, random_state=4)
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, LogisticRegression)
