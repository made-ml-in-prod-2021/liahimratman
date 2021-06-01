from dataclasses import dataclass
import pickle
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


def load_saved_transformers(column_transformer_save_path: str, scaler_transformer_save_path: str):
    """
    Load saved prefitted transformers
    :param column_transformer_save_path: saved column transformer path
    :param scaler_transformer_save_path: saved custom standard scaler transformer path
    :return: dict with transformers
    """
    with open(column_transformer_save_path, 'rb') as output_stream:
        column_transformer = pickle.load(output_stream)
    with open(scaler_transformer_save_path, 'rb') as output_stream:
        scaler_transformer = pickle.load(output_stream)
    transformers = {
        "column_transformer": column_transformer,
        "standard_scaler_transformer": scaler_transformer,
    }

    return transformers
