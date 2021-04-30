import logging
import sys

import click

from ml_project.data_functions.make_dataset import read_data
from ml_project.models.model_fit_predict import save_predictions, load_model
from ml_project.params.evaluation_pipeline_params import (
    EvaluationPipelineParams,
    read_evaluation_pipeline_params,
)

from ml_project.features.build_features import build_transformers, make_features
from ml_project.models.model_fit_predict import predict_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(evaluation_pipeline_params: EvaluationPipelineParams):
    logger.info(f"start evaluation pipeline with params {evaluation_pipeline_params}")

    logger.info("Reading data ...")
    data = read_data(evaluation_pipeline_params.input_data_path)
    logger.info(f"Input data shape is {data.shape}")

    logger.info("Building transformers ...")
    transformers = build_transformers(evaluation_pipeline_params.feature_params)
    logger.info("Transformers built")

    logger.info("Making features ...")
    eval_features, _ = make_features(transformers, data, mode="eval",
                                  scaler_params=evaluation_pipeline_params.scaler_params)
    logger.info(f"Features shape is {eval_features.shape}")

    logger.info("Loading model ...")
    model = load_model(evaluation_pipeline_params.input_model_path)
    logger.info("Model loaded")

    logger.info("Start making predictions ...")
    predicts = predict_model(
        model,
        eval_features
    )
    logger.info(f"{len(predicts)} predictions made")

    logger.info("Saving predictions ...")
    save_predictions(evaluation_pipeline_params.output_data_path, predicts)
    logger.info("Predictions saved")
    logger.info("Evaluation pipeline ended")

    return evaluation_pipeline_params.output_data_path


@click.command(name="predict_pipeline")
@click.argument("config_path")
def evaluation_pipeline_command(config_path: str):
    params = read_evaluation_pipeline_params(config_path)
    predict_pipeline(params)


if __name__ == "__main__":
    evaluation_pipeline_command()
