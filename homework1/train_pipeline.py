import logging
import sys
import hydra
from hydra.utils import to_absolute_path

from ml_project.data_functions.make_dataset import read_data, split_train_val_data
from ml_project.models.model_fit_predict import save_metrics, train_model, \
    serialize_model, predict_model, evaluate_model
from ml_project.params.evaluation_pipeline_params import write_evaluation_pipeline_params
from ml_project.params.train_pipeline_params import TrainingPipelineParams, \
    read_training_pipeline_params
from ml_project.features.build_features import make_features, extract_target, build_transformers


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    """
    Training pipeline.
    :param training_pipeline_params:
    parameters of feature extraction, splitting, training and saving
    :return: None
    """
    logger.info("Start train pipeline. Pipeline params:")
    logger.info(training_pipeline_params)

    logger.info("Reading data ...")
    data = read_data(to_absolute_path(training_pipeline_params.input_data_path))
    logger.info("Data shape:")
    logger.info(data.shape)

    logger.info("Splitting data ...")
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info("train_df.shape:")
    logger.info(train_df.shape)
    logger.info("val_df.shape:")
    logger.info(val_df.shape)

    logger.info("Building transformers ...")
    transformers = build_transformers(training_pipeline_params.feature_params)
    logger.info("Transformers built")

    logger.info("Making train features ...")
    train_features, transformers = make_features(transformers, train_df, mode="train")
    logger.info("Train features shape:")
    logger.info(train_features.shape)

    logger.info("Extracting train target ...")
    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    logger.info("Training target extracted")

    logger.info("Training ...")
    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )
    logger.info("Model trained")

    logger.info("Making validation features ...")
    val_features, _ = make_features(transformers, val_df, mode="val")
    logger.info("Val_features.shape:")
    logger.info(val_features.shape)

    logger.info("Extracting validation target ...")
    val_target = extract_target(val_df, training_pipeline_params.feature_params)
    logger.info("Validation target extracted")

    logger.info("Start making predictions ...")
    predicts = predict_model(
        model,
        val_features
    )
    logger.info("Predictions made")

    logger.info("Start evaluating metrics ...")
    metrics = evaluate_model(
        predicts,
        val_target
    )
    logger.info("Metrics:")
    logger.info(metrics)

    logger.info("Saving metrics ...")
    save_metrics(to_absolute_path(training_pipeline_params.metric_path), metrics)
    logger.info("Metrics saved")

    logger.info("Saving model ...")
    serialize_model(model,
                                    to_absolute_path(training_pipeline_params.output_model_path))
    logger.info("Model saved")

    logger.info("Saving evaluation config ...")
    eval_config_path = write_evaluation_pipeline_params(
        output_path=to_absolute_path(training_pipeline_params.output_config_path),
        path_to_model=training_pipeline_params.output_model_path,
        column_save_path=training_pipeline_params.column_transformer_save_path,
        scaler_save_path=training_pipeline_params.scaler_transformer_save_path,
        feature_params=training_pipeline_params.feature_params,
        transformers=transformers
    )
    logger.info("Evaluation config saved")
    logger.info("Training pipeline ended")

    return metrics, eval_config_path


@hydra.main()
def train_app(cfg: dict) -> None:
    """
    Running train app.
    :param cfg: Training config. By default use configs/train_config.yaml
    :return: None
    """
    params = read_training_pipeline_params(cfg)
    train_pipeline(params)


if __name__ == "__main__":
    train_app()
