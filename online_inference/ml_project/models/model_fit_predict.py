import pickle
from typing import Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


SklearnClassificationModel = Union[RandomForestClassifier, LogisticRegression]


def load_model(input_model_path: str) -> SklearnClassificationModel:
    """
    Load model
    :param input_model_path: saved model path
    :return: SklearnClassificationModel
    """
    with open(input_model_path, 'rb') as input_stream:
        model = pickle.load(input_stream)

    return model
