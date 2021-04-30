import json
import logging
import sys

import click

from ml_project.data_functions import read_data, split_train_val_data
from ml_project.params.evaluation_pipeline_params import write_evaluation_pipeline_params
from ml_project.params.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from ml_project.features import make_features
from ml_project.features.build_features import extract_target, build_transformers
from ml_project.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    transformers = build_transformers(training_pipeline_params.feature_params)
    train_features, transformers = make_features(transformers, train_df, mode="train")

    train_target = extract_target(train_df, training_pipeline_params.feature_params)

    logger.info(f"train_features.shape is {train_features.shape}")

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    val_features, _ = make_features(transformers, val_df, mode="eval")
    val_target = extract_target(val_df, training_pipeline_params.feature_params)

    logger.info(f"val_features.shape is {val_features.shape}")
    predicts = predict_model(
        model,
        val_features
    )

    metrics = evaluate_model(
        predicts,
        val_target
    )

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)

    logger.info(f"metrics is {metrics}")

    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)

    eval_config_path = write_evaluation_pipeline_params(output_path="new_eval_config.yaml", path_to_model=path_to_model,
                                                        feature_params=training_pipeline_params.feature_params,
                                                        scaler=transformers["standard_scaler_transformer"])
    # import yaml
    # with open("new_eval_config.yaml", "r") as input_stream:
    #     print(yaml.safe_load(input_stream))
    return metrics, eval_config_path


# @click.command(name="train_pipeline")
# @click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command(
        "C:/Users/Mikhail Korotkov/PycharmProjects/liahimratman/ml_project/configs/train_config.yaml")
