import os
import click
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.argument("splitted-dir")
@click.argument("model-dir")
def train_model(splitted_dir: str, model_dir: str):
    model = LogisticRegression(random_state=42)
    df = pd.read_csv(os.path.join(splitted_dir, "train.csv"), index_col=0)
    X_train = df.drop(["target"], axis=1, inplace=False).values
    y_train = df["target"].values

    model.fit(X_train, y_train)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train_model()
