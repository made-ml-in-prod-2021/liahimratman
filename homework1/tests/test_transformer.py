from typing import List

import pandas as pd

from ml_project.params.feature_params import FeatureParams
from ml_project.features.build_features import make_features, build_transformers


def test_transformers(
    make_fake_dataset: str, categorical_features: List[str], numerical_features: List[str]
):
    """
    Test custom transformers
    :param make_fake_dataset: fake dataset path
    :param categorical_features: categorical_features
    :param numerical_features: numerical_features
    :return: None
    """
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col="target",
    )
    data = pd.read_csv(make_fake_dataset)
    transformers = build_transformers(params)
    features, _ = make_features(transformers, data, mode="train")

    assert features.values.mean(axis=0).max() < 1e-9
    assert features.values.mean(axis=0).min() > -1e-9
    assert features.values.std(axis=0).max() < 1 + 1e-9
    assert features.values.std(axis=0).min() > 1 - 1e-9
