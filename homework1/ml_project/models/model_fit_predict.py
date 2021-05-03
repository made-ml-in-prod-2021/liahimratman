import json
import pickle
from typing import Dict, Union
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, jaccard_score, f1_score, roc_auc_score
from ml_project.params.train_params import TrainingParams


SklearnClassificationModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnClassificationModel:
    """
    Train model
    :param features: preprocessed data
    :param target: targets
    :param train_params: training parameters
    :return: SklearnClassificationModel
    """
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            max_iter=train_params.max_iter,
            random_state=train_params.random_state
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)

    return model


def predict_model(
        model: SklearnClassificationModel, features: pd.DataFrame
) -> np.ndarray:
    """
    Make predictions
    :param model: SklearnClassificationModel
    :param features: preprocessed data
    :return: predictions (np.ndarray)
    """
    predicts = model.predict(features)

    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model metrics (Accuracy, Precision, Recall, Jaccard, f1, ROC_AUC_SCORE)
    :param predicts: predictions
    :param target: targets
    :return: Dict[str, float]
    """

    return {
        "accuracy": accuracy_score(target, predicts),
        "precision": precision_score(target, predicts),
        "recall": recall_score(target, predicts),
        "jaccard": jaccard_score(target, predicts),
        "f1": f1_score(target, predicts),
        "roc_auc_score": roc_auc_score(target, predicts),
    }


def serialize_model(model: SklearnClassificationModel, output: str) -> str:
    """
    Save model
    :param model: SklearnClassificationModel
    :param output: output path
    :return: output path
    """
    with open(output, "wb") as output_stream:
        pickle.dump(model, output_stream)

    return output


def load_model(input_model_path: str) -> SklearnClassificationModel:
    """
    Load model
    :param input_model_path: saved model path
    :return: SklearnClassificationModel
    """
    with open(input_model_path, 'rb') as input_stream:
        model = pickle.load(input_stream)

    return model


def save_predictions(output_data_path: str, predicts: list) -> str:
    """
    Saved model predictions
    :param output_data_path: predictions path
    :param predicts: predictions
    :return: predictions path
    """
    pd.DataFrame(predicts, columns=['target']).to_csv(output_data_path, index_label='index')

    return output_data_path


def save_metrics(metric_path: str, metrics: dict) -> str:
    """
    Save model training and validation metrics
    :param metric_path: metrics path
    :param metrics: metrics
    :return: metrics path
    """
    with open(metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)

    return metric_path
