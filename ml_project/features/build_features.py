import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_project.params.feature_params import FeatureParams
from ml_project.transformers.make_transformers import StandardScalerTransformer


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),]
    )
    return num_pipeline


def make_features(transformers: dict, df: pd.DataFrame, mode="val", scaler_params=None) -> pd.DataFrame:
    if mode in ["train", "eval"]:
        transformers["column_transformer"].fit(df)
    df = pd.DataFrame(transformers["column_transformer"].transform(df))
    if mode == "train":
        transformers["standard_scaler_transformer"].fit(df)
    elif mode == "eval":
        precomputed_means = list(map(float, scaler_params.mean))
        precomputed_scales = list(map(float, scaler_params.scale))
        transformers["standard_scaler_transformer"].fit(df, precomputed_means, precomputed_scales)
    return pd.DataFrame(transformers["standard_scaler_transformer"].transform(df)), transformers


def build_transformers(params: FeatureParams) -> dict:
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


# def build_standard_scaler_transformer() -> StandardScalerTransformer:
#     transformer = StandardScalerTransformer()
#     return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target
