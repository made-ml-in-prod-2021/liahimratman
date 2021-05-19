import os
import pickle
import json
from typing import List, Tuple
from pathlib import Path
import pandas as pd
from py._path.local import LocalPath
from sklearn.linear_model import LogisticRegression
from faker import Faker
import pytest

from ml_project.data_functions.make_dataset import split_train_val_data
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
def fake_dataset_path():
    return "tests_data/fake_dataset.csv"


@pytest.fixture()
def target_col():
    return "target"


@pytest.fixture()
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


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


def test_load_dataset(dataset_path: str, target_col: str):
    """
    Dataset loading test
    :param dataset_path: dataset path
    :param target_col: target
    :return: None
    """
    data = pd.read_csv(dataset_path)
    assert len(data) > 10
    assert target_col in data.keys()


def test_split_dataset(dataset_path: str):
    """
    Dataset split test
    :param dataset_path: dataset path
    :return: None
    """
    val_size = 0.2
    splitting_params = SplittingParams(random_state=239, val_size=val_size,)
    data = pd.read_csv(dataset_path)
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] > 10
    assert val.shape[0] > 10


@pytest.fixture()
def features_and_target(
    dataset_path: str, categorical_features: List[str], numerical_features: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target from dataset
    :param dataset_path: dataset path
    :param categorical_features: categorical_features
    :param numerical_features: numerical_features
    :return: Tuple[pd.DataFrame, pd.Series]
    """
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col="target",
    )
    data = pd.read_csv(dataset_path)
    transformers = build_transformers(params)
    features, _ = make_features(transformers, data, mode="train")
    target = extract_target(data, params)

    return features, target


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    """
    Test model training
    :param features_and_target: Tuple[pd.DataFrame, pd.Series]
    :return: None
    """
    features, target = features_and_target
    model = train_model(features, target,
                        train_params=TrainingParams(model_type="LogisticRegression",
                                                    random_state=4,
                                                    max_iter=1000))
    assert isinstance(model, LogisticRegression)
    assert model.predict(features).shape[0] == target.shape[0]


def test_serialize_model(tmpdir: LocalPath):
    """
    Test model serialization
    :param tmpdir: tmpdir
    :return: None
    """
    expected_output = tmpdir.join("model.pkl")
    model = LogisticRegression(max_iter=1000, random_state=4)
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, "rb") as output_stream:
        model = pickle.load(output_stream)
    assert isinstance(model, LogisticRegression)


def get_feature_config(
        target_column: str,
        categorical_features: List[str],
        numerical_features: List[str],
) -> FeatureParams:
    """
    Get feature config
    :param target_column: target
    :param categorical_features: categorical_features
    :param numerical_features: numerical_features
    :return: FeatureParams config
    """
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
) -> Tuple[str, str]:
    """
    Full training pipeline
    :param categorical_features: categorical_features
    :param dataset_path: dataset path
    :param numerical_features: numerical_features
    :param target_col: target
    :param tmpdir: tmpdir
    :return: metrics_path, model_save_path
    """
    model_save_path = str(tmpdir / "model.pkl")
    columns_transformer_save_path = str(tmpdir / "columns.pkl")
    scaler_transformer_save_path = str(tmpdir / "scaler.pkl")
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
            max_iter=1000,
        ),
        input_data_path=dataset_path,
        output_model_path=model_save_path,
        output_config_path=output_config_path,
        metric_path=metrics_path,
        column_transformer_save_path=columns_transformer_save_path,
        scaler_transformer_save_path=scaler_transformer_save_path,
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
    """
    Test full pipeline
    :param tmpdir: tmpdir
    :param dataset_path: dataset path
    :param categorical_features: categorical_features
    :param numerical_features: numerical_features
    :param target_col: target
    :return: None
    """
    metrics_path, model_save_path = train_tmp_model(categorical_features, dataset_path,
                                                                    numerical_features, target_col,
                                                                    tmpdir)
    assert Path(model_save_path).exists()
    assert Path(metrics_path).exists()
    with open(metrics_path, "r") as input_stream:
        metric_values = json.load(input_stream)
        assert metric_values["accuracy"] > 0
        assert metric_values["f1"] > 0
        assert metric_values["precision"] > 0
        assert metric_values["recall"] > 0


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


def test_train_pipeline_on_fake_dataset(
        tmpdir: Path,
        fake_dataset_path: str,
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str
):
    """
    Test ful pipeline on fake dataset
    :param tmpdir: tmpdir
    :param fake_dataset_path: fake dataset path
    :param categorical_features: categorical_features
    :param numerical_features: numerical_features
    :param target_col: target
    :return: None
    """
    make_fake_dataset(fake_dataset_path)
    metrics_path, model_save_path = train_tmp_model(categorical_features, fake_dataset_path,
                                                                    numerical_features, target_col,
                                                                    tmpdir)
    assert Path(model_save_path).exists()
    assert Path(metrics_path).exists()
    with open(metrics_path, "r") as input_stream:
        metric_values = json.load(input_stream)
        assert metric_values["accuracy"] > 0
        assert metric_values["f1"] > 0
        assert metric_values["precision"] > 0
        assert metric_values["recall"] > 0


def test_transformers(
    fake_dataset_path: str, categorical_features: List[str], numerical_features: List[str]
):
    """
    Test custom transformers
    :param fake_dataset_path: fake dataset path
    :param categorical_features: categorical_features
    :param numerical_features: numerical_features
    :return: None
    """
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col="target",
    )
    make_fake_dataset(fake_dataset_path)
    data = pd.read_csv(fake_dataset_path)
    transformers = build_transformers(params)
    features, _ = make_features(transformers, data, mode="train")

    assert features.values.mean(axis=0).max() < 1e-9
    assert features.values.mean(axis=0).min() > -1e-9
    assert features.values.std(axis=0).max() < 1 + 1e-9
    assert features.values.std(axis=0).min() > 1 - 1e-9
