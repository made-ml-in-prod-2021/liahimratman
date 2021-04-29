from dataclasses import dataclass
from .feature_params import FeatureParams
from .scaler_params import ScalerParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class EvaluationPipelineParams:
    input_data_path: str
    input_model_path: str
    output_data_path: str
    output_metric_path: str
    feature_params: FeatureParams
    scaler_params: ScalerParams


EvaluationPipelineParamsSchema = class_schema(EvaluationPipelineParams)


def read_evaluation_pipeline_params(path: str) -> EvaluationPipelineParams:
    with open(path, "r") as input_stream:
        schema = EvaluationPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
