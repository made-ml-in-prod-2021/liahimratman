import os
import pickle
import json
from typing import Tuple
from pathlib import Path
import pandas as pd
from py._path.local import LocalPath
from sklearn.linear_model import LogisticRegression

from ml_project.data_functions.make_dataset import split_train_val_data
from ml_project.params.split_params import SplittingParams
from ml_project.params.train_params import TrainingParams
from ml_project.models.model_fit_predict import train_model, serialize_model


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


def test_train_pipeline(
        train_tmp_model: Tuple,
):
    """
    Test full pipeline
    :param train_tmp_model: metrics_path, model_save_path
    :return: None
    """
    metrics_path, model_save_path = train_tmp_model#(dataset_path, feature_config, tmpdir)
    assert Path(model_save_path).exists()
    assert Path(metrics_path).exists()
    with open(metrics_path, "r") as input_stream:
        metric_values = json.load(input_stream)
        assert metric_values["accuracy"] > 0
        assert metric_values["f1"] > 0
        assert metric_values["precision"] > 0
        assert metric_values["recall"] > 0


def test_train_pipeline_on_fake_dataset(
        train_fake_model: Tuple,
):
    """
    Test full pipeline on fake dataset
    :param train_fake_model: metrics_path, model_save_path
    :return: None
    """
    metrics_path, model_save_path = train_fake_model
    assert Path(model_save_path).exists()
    assert Path(metrics_path).exists()
    with open(metrics_path, "r") as input_stream:
        metric_values = json.load(input_stream)
        assert metric_values["accuracy"] > 0
        assert metric_values["f1"] > 0
        assert metric_values["precision"] > 0
        assert metric_values["recall"] > 0


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
