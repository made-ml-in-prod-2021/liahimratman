import logging
import sys
import hydra
from hydra.utils import to_absolute_path

from ml_project.data_functions.make_dataset import read_data
from ml_project.models.model_fit_predict import save_predictions, load_model
from ml_project.params.evaluation_pipeline_params import (
    EvaluationPipelineParams,
    read_evaluation_pipeline_params,
    load_saved_transformers,
)
from ml_project.features.build_features import make_features
from ml_project.models.model_fit_predict import predict_model


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(evaluation_pipeline_params: EvaluationPipelineParams):
    """
    Evaluation pipiline.
    :param evaluation_pipeline_params: parameters of feature extraction and model evaluation
    :return: None
    """
    logger.info(f"start evaluation pipeline with params {evaluation_pipeline_params}")

    logger.info("Reading data ...")
    data = read_data(to_absolute_path(evaluation_pipeline_params.input_data_path))
    logger.info("Input data shape:")
    logger.info(data.shape)

    logger.info("Building transformers ...")
    transformers = load_saved_transformers(
        column_transformer_save_path=
        to_absolute_path(evaluation_pipeline_params.column_transformer_save_path),
        scaler_transformer_save_path=
        to_absolute_path(evaluation_pipeline_params.scaler_transformer_save_path)
    )
    logger.info("Transformers built")

    logger.info("Making features ...")
    eval_features, _ = make_features(transformers, data, mode="val")
    logger.info("Features shape:")
    logger.info(eval_features.shape)

    logger.info("Loading model ...")
    model = load_model(to_absolute_path(evaluation_pipeline_params.input_model_path))
    logger.info("Model loaded")

    logger.info("Start making predictions ...")
    predicts = predict_model(
        model,
        eval_features
    )
    logger.info("Predictions made")

    logger.info("Saving predictions ...")
    save_predictions(to_absolute_path(evaluation_pipeline_params.output_data_path), predicts)
    logger.info("Predictions saved")
    logger.info("Evaluation pipeline ended")

    return evaluation_pipeline_params.output_data_path


@hydra.main()
def predict_app(cfg) -> None:
    """
    Running prediction app.
    :param cfg: Evaluation config. By default use configs/evaluation_config.yaml
    :return: None
    """
    params = read_evaluation_pipeline_params(cfg)
    predict_pipeline(params)


if __name__ == "__main__":
    predict_app()
