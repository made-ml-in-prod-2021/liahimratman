from dataclasses import dataclass
from .split_params import SplittingParams
from .feature_params import FeatureParams
from .train_params import TrainingParams
from marshmallow_dataclass import class_schema


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    column_transformer_save_path: str
    scaler_transformer_save_path: str
    output_config_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(params) -> TrainingPipelineParams:
    schema = TrainingPipelineParamsSchema()
    return schema.load(params)
