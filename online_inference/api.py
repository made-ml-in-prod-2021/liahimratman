import logging
from typing import List, Dict

from fastapi import FastAPI, HTTPException, Request
import numpy as np
import pandas as pd
import yaml

from ml_project.features.build_features import make_features
from ml_project.models.model_fit_predict import load_model
from ml_project.params.evaluation_pipeline_params import load_saved_transformers
from online_inference.schemas import DiseaseModel, DiseaseResponseModel, Settings


logger = logging.getLogger(__name__)
settings = Settings()


def rebuild_dataframe(params: DiseaseModel, metadata: Dict[str, np.dtype]) -> pd.DataFrame:
    try:
        data = pd.DataFrame(params.features, columns=params.columns)
    except ValueError:
        error_msg = "Failed to construct DataFrame from passed data"
        logger.exception(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    for key in metadata['feature_params']['numerical_features']:
        if key not in data.columns:
            logger.error(f"Column {key} not found in data")
            raise HTTPException(status_code=400)
        if data[key].dtype != "float64":
            try:
                data[key] = data[key].astype("float64")
            except ValueError:
                logger.exception(f"Failed to cast column {key} to dtype float64")
                raise HTTPException(status_code=400)
    for key in metadata['feature_params']['categorical_features']:
        if key not in data.columns:
            logger.error(f"Column {key} not found in data")
            raise HTTPException(status_code=400)
        if data[key].dtype != "int64":
            try:
                data[key] = data[key].astype("int64")
            except ValueError:
                logger.exception(f"Failed to cast column {key} to dtype int64")
                raise HTTPException(status_code=400)

    input_columns = metadata['feature_params']['categorical_features'] + \
                    metadata['feature_params']['numerical_features'] + ['target']
    if "target" not in data:
        data["target"] = 0

    return data[input_columns]


app = FastAPI()


@app.get("/")
def main():
    return "it's an entry point of our predictor"


@app.on_event("startup")
def load_artifacts():
    logger.info("Started loading model, transformers and metadata")
    app.state.transformers = load_saved_transformers(
        column_transformer_save_path=
        str(settings.column_transformer_save_path),
        scaler_transformer_save_path=
        str(settings.scaler_transformer_save_path)
    )
    app.state.model = load_model(str(settings.model_path))
    with open(str(settings.metadata_path), "rb") as input_stream:
        app.state.metadata = yaml.safe_load(input_stream)
    logger.info("Ended loading model, transformers and metadata")


@app.get("/is_ready")
def health() -> bool:
    return app.state.model is not None


@app.post("/predict", response_model=List[DiseaseResponseModel])
def predict(request: Request, params: DiseaseModel):
    data = rebuild_dataframe(params, app.state.metadata)
    eval_features, _ = make_features(app.state.transformers, data, mode="val")
    predictions = request.app.state.model.predict(eval_features)
    return [
        DiseaseResponseModel(id=id_, target=pred == 1) for id_, pred in zip(params.ids, predictions)
    ]
