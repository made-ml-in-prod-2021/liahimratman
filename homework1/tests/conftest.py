from typing import List, Tuple
from pathlib import Path
import pandas as pd
from faker import Faker
import pytest

from ml_project.params.split_params import SplittingParams
from ml_project.params.train_params import TrainingParams
from ml_project.params.train_pipeline_params import TrainingPipelineParams
from ml_project.params.feature_params import FeatureParams
from ml_project.features.build_features import make_features, extract_target, build_transformers
from train_pipeline import train_pipeline


@pytest.fixture(scope='module')
def dataset_path():
    return "tests_data/train_data_sample.csv"


@pytest.fixture(scope='module')
def fake_dataset_path():
    return "tests_data/fake_dataset.csv"


@pytest.fixture(scope='module')
def target_col():
    return "target"


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
def make_fake_dataset(fake_dataset_path: str):
    """
    Make and save fake dataset
    :param fake_dataset_path: fake dataset saving path
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

    with open(fake_dataset_path, 'w') as input_stream:
        input_stream.write(fake_data)

    return fake_dataset_path


@pytest.fixture(scope='module')
def feature_config(target_col: str, categorical_features: List[str], numerical_features: List[str]) -> FeatureParams:
    """
    Get feature config
    :param target_col: target
    :param categorical_features: categorical_features
    :param numerical_features: numerical_features
    :return: FeatureParams config
    """
    config = FeatureParams(
        target_col=target_col,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    return config


@pytest.fixture(scope='function')
def train_tmp_model(
        dataset_path: str,
        feature_config: FeatureParams,
        tmpdir: Path
) -> Tuple[str, str]:
    """
    Full training pipeline
    :param feature_config: features config
    :param dataset_path: dataset path
    :param tmpdir: tmpdir
    :return: metrics_path, model_save_path
    """
    model_save_path = str(tmpdir / "model.pkl")
    columns_transformer_save_path = str(tmpdir / "columns.pkl")
    scaler_transformer_save_path = str(tmpdir / "scaler.pkl")
    metrics_path = str(tmpdir / "metrics.json")
    output_config_path = str(tmpdir / "out_config.yaml")
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


@pytest.fixture(scope='function')
def train_fake_model(
        make_fake_dataset: str,
        feature_config: FeatureParams,
        tmpdir: Path
) -> Tuple[str, str]:
    """
    Full training pipeline
    :param feature_config: features config
    :param make_fake_dataset: fake dataset path
    :param tmpdir: tmpdir
    :return: metrics_path, model_save_path
    """
    model_save_path = str(tmpdir / "model.pkl")
    columns_transformer_save_path = str(tmpdir / "columns.pkl")
    scaler_transformer_save_path = str(tmpdir / "scaler.pkl")
    metrics_path = str(tmpdir / "metrics.json")
    output_config_path = str(tmpdir / "out_config.yaml")
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
        input_data_path=make_fake_dataset,
        output_model_path=model_save_path,
        output_config_path=output_config_path,
        metric_path=metrics_path,
        column_transformer_save_path=columns_transformer_save_path,
        scaler_transformer_save_path=scaler_transformer_save_path,
    )
    train_pipeline(config)

    return metrics_path, model_save_path