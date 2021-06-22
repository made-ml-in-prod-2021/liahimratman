import os
import click
import pickle

import numpy as np
import pandas as pd


@click.command("predict")
@click.argument("input-dir")
@click.argument("model-dir")
@click.argument("output-dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"), index_col=0)
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    X = scaler.transform(data)
    prediction = model.predict(X).astype(np.int)

    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, "prediction.csv"), prediction)


if __name__ == "__main__":
    predict()
