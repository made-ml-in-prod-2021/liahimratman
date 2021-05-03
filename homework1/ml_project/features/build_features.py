import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_project.params.feature_params import FeatureParams
from ml_project.transformers.make_transformers import StandardScalerTransformer


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process categorical features
    :param categorical_df: categorical data
    :return: transformed data
    """
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_categorical_pipeline() -> Pipeline:
    """
    Build categorical pipeline
    :return: Pipeline
    """
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process numerical features
    :param numerical_df: numerical data
    :return: transformed data
    """
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline() -> Pipeline:
    """
    Build numerical pipeline
    :return: Pipeline
    """
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="mean")), ]
    )
    return num_pipeline


def make_features(transformers: dict, data: pd.DataFrame, mode="val") -> pd.DataFrame:
    """
    Make features
    :param transformers: Dict with column transformer and custom standard scaler transformer
    :param data: input data
    :param mode: processing mode
    :return: pd.DataFrame (preprocessed data)
    """
    if mode in ["train"]:
        transformers["column_transformer"].fit(data)

    data = pd.DataFrame(transformers["column_transformer"].transform(data))

    if mode == "train":
        transformers["standard_scaler_transformer"].fit(data)

    return pd.DataFrame(transformers["standard_scaler_transformer"].transform(data)), transformers


def build_transformers(params: FeatureParams) -> {str: ColumnTransformer,
                                                  str: StandardScalerTransformer}:
    """
    Building transformers
    :param params: FeatureParams
    :return: Dict[str, transformer]
    """
    column_transformer = ColumnTransformer(
        [
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
        ]
    )

    standard_scaler_transformer = StandardScalerTransformer()
    transformers = {
        "column_transformer": column_transformer,
        "standard_scaler_transformer": standard_scaler_transformer,
    }

    return transformers


def extract_target(data: pd.DataFrame, params: FeatureParams) -> pd.Series:
    """
    Extract target from data
    :param data: imput data
    :param params: FeatureParams
    :return: pd.Series
    """
    target = data[params.target_col]

    return target
