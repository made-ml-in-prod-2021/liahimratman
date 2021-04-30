from dataclasses import dataclass

from transformers.make_transformers import StandardScalerTransformer
from .feature_params import FeatureParams
from .scaler_params import ScalerParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class EvaluationPipelineParams:
    input_data_path: str
    output_data_path: str
    input_model_path: str
    feature_params: FeatureParams
    scaler_params: ScalerParams


EvaluationPipelineParamsSchema = class_schema(EvaluationPipelineParams)


def read_evaluation_pipeline_params(path: str) -> EvaluationPipelineParams:
    with open(path, "r") as input_stream:
        schema = EvaluationPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))


def write_evaluation_pipeline_params(output_path: str, path_to_model: str,
                                     feature_params: FeatureParams, scaler: StandardScalerTransformer) -> str:
    eval_config = {
        "input_data_path": None,
        "output_data_path": "output_data/predicted.csv",
        "input_model_path": path_to_model,
        "scaler_params": {
            "mean": list(map(str, list(scaler.mean_))),
            "scale": list(map(str, list(scaler.scale_))),
        },
        "feature_params": {
            "numerical_features": feature_params.numerical_features,
            "categorical_features": feature_params.categorical_features,
        }
    }

    with open(output_path, "w") as output_stream:
        yaml.safe_dump(eval_config, output_stream)

    return output_path
