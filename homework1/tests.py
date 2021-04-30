import os
import pytest
import pickle
from typing import List, Tuple
import pandas as pd
from py._path.local import LocalPath
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import json

from ml_project.data_functions.make_dataset import read_data, split_train_val_data
from ml_project.params.split_params import SplittingParams
from ml_project.params.train_params import TrainingParams
from ml_project.params.train_pipeline_params import TrainingPipelineParams
from ml_project.params.feature_params import FeatureParams
from ml_project.features.build_features import make_features, extract_target, build_transformers
from ml_project.models.model_fit_predict import train_model, serialize_model
from train_pipeline import train_pipeline

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


def get_feature_config(
        target_column: str,
        categorical_features: List[str],
        numerical_features: List[str],
):
    config = FeatureParams(
        target_col=target_column,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )
    return config


def train_tmp_model(
        categorical_features: List[str],
        dataset_path: str,
        numerical_features: List[str],
        target_col: str,
        tmpdir: Path
):
    model_save_path = str(tmpdir / "model.pkl")
    metrics_path = str(tmpdir / "metrics.json")
    output_config_path = str(tmpdir / "out_config.yaml")
    feature_config = get_feature_config(target_col, categorical_features, numerical_features)
    config = TrainingPipelineParams(
        splitting_params=SplittingParams(
            val_size=0.2,
            random_state=4,
        ),
        feature_params=feature_config,
        train_params=TrainingParams(
            model_type="LogisticRegression",
            random_state=4,
        ),
        input_data_path=dataset_path,
        output_model_path=model_save_path,
        output_config_path=output_config_path,
        metric_path=metrics_path,
    )
    train_pipeline(config)
    return metrics_path, model_save_path


def test_train_pipeline(
        tmpdir: Path,
        dataset_path: str,
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str
):
    metrics_path, model_save_path = train_tmp_model(categorical_features, dataset_path,
                                                                    numerical_features, target_col,
                                                                    tmpdir)
    assert Path(model_save_path).exists()
    assert Path(metrics_path).exists()
    with open(metrics_path, "r") as f:
        metric_values = json.load(f)
        assert metric_values["accuracy"] > 0
        assert metric_values["f1"] > 0
        assert metric_values["precision"] > 0
        assert metric_values["recall"] > 0
