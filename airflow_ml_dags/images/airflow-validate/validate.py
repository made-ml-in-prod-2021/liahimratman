import os
import pickle
import json
import click

import pandas as pd
from sklearn.metrics import f1_score


@click.command("validate")
@click.argument("splitted-dir")
@click.argument("model-dir")
def validate(splitted_dir: str, model_dir: str):
    # Loading data
    val = pd.read_csv(os.path.join(splitted_dir, "val.csv"), index_col=0)
    X_val = val.drop(["target"], axis=1, inplace=False).values
    y_val = val["target"].values

    # Loading and scoring model
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    prediction = model.predict(X_val)
    accuracy = model.score(X_val, y_val)
    f1 = f1_score(y_val, prediction, average='micro')

    # Saving results
    scores = {"Accuracy": accuracy, "f1": f1}
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "scores.json"), "w") as f:
        json.dump(scores, f)


if __name__ == "__main__":
    validate()
