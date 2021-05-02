from dataclasses import dataclass
import pickle
import yaml
from hydra.utils import to_absolute_path
from marshmallow_dataclass import class_schema

from .feature_params import FeatureParams


@dataclass()
class EvaluationPipelineParams:
    input_data_path: str
    output_data_path: str
    input_model_path: str
    column_transformer_save_path: str
    scaler_transformer_save_path: str
    feature_params: FeatureParams


EvaluationPipelineParamsSchema = class_schema(EvaluationPipelineParams)


def read_evaluation_pipeline_params(params: dict) -> EvaluationPipelineParams:
    schema = EvaluationPipelineParamsSchema()
    return schema.load(params)


def write_evaluation_pipeline_params(output_path: str, path_to_model: str, column_save_path: str,
                                     scaler_save_path: str,
                                     feature_params: FeatureParams, transformers: dict) -> str:
    eval_config = {
        "input_data_path": None,
        "output_data_path": "output_data/predicted.csv",
        "input_model_path": path_to_model,
        "column_transformer_save_path": column_save_path,
        "scaler_transformer_save_path": scaler_save_path,
        "feature_params": {
            "numerical_features": feature_params.numerical_features,
            "categorical_features": feature_params.categorical_features,
        }
    }

    with open(to_absolute_path(column_save_path), 'wb') as output_transformers_stream:
        pickle.dump(transformers["column_transformer"], output_transformers_stream)

    with open(to_absolute_path(scaler_save_path), 'wb') as output_transformers_stream:
        pickle.dump(transformers["standard_scaler_transformer"], output_transformers_stream)

    with open(output_path, "w") as output_stream:
        yaml.safe_dump(eval_config, output_stream)

    return output_path


def load_saved_transformers(column_transformer_save_path: str, scaler_transformer_save_path: str):
    with open(column_transformer_save_path, 'rb') as output_stream:
        column_transformer = pickle.load(output_stream)
    with open(scaler_transformer_save_path, 'rb') as output_stream:
        scaler_transformer = pickle.load(output_stream)
    transformers = {
        "column_transformer": column_transformer,
        "standard_scaler_transformer": scaler_transformer,
    }

    return transformers
