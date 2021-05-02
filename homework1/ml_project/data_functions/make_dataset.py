# -*- coding: utf-8 -*-
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_project.params.split_params import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    """
    Read input data
    :param path: input data path
    :return: data
    """
    data = pd.read_csv(path)

    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data on training and validation dataset
    :param data: input data
    :param params: splitting parameters
    :return: Tuple[pd.DataFrame, pd.DataFrame]
    """
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )

    return train_data, val_data
