from typing import Union, List
# from pathlib import Path

from pydantic import BaseModel, BaseSettings


class DiseaseModel(BaseModel):
    ids: List[int]
    features: List[List[Union[int, float]]]
    columns: List[str]


class DiseaseResponseModel(BaseModel):
    id: int
    target: bool


class Settings(BaseSettings):
    model_path: str = 'ml_project/models/model.pkl'
    metadata_path: str = 'configs/evaluation_config.yaml'
    column_transformer_save_path: str = 'ml_project/models/custom_column_transformers.pkl'
    scaler_transformer_save_path: str = 'ml_project/models/custom_scaler_transformers.pkl'
