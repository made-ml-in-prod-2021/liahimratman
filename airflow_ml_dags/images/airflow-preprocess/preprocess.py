import os
import pickle
import click

import pandas as pd
from sklearn.preprocessing import StandardScaler


@click.command("preprocess")
@click.argument("input-dir")
@click.argument("output-dir")
@click.argument("model-dir")
def preprocess(input_dir: str, output_dir: str, model_dir: str):
    # Loading data
    target = pd.read_csv(os.path.join(input_dir, "target.csv"), index_col=0)
    features = pd.read_csv(os.path.join(input_dir, "data.csv"), index_col=0)
    feature_columns = features.columns.tolist()

    # Preprocess data
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = pd.DataFrame(features)
    features.columns = feature_columns

    # Saving processed data
    df = features.merge(target, right_index=True, left_index=True)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "processed.csv"))

    # Saving scaler
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    preprocess()
