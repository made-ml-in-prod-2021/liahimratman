import os
import click

import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("split")
@click.argument("processed-dir")
@click.argument("splitted-dir")
def split(processed_dir: str, splitted_dir: str):
    df = pd.read_csv(os.path.join(processed_dir, "processed.csv"), index_col=0)
    train, val = train_test_split(df, random_state = 42)
    os.makedirs(splitted_dir, exist_ok=True)
    train.to_csv(os.path.join(splitted_dir, "train.csv"))
    val.to_csv(os.path.join(splitted_dir, "val.csv"))


if __name__ == "__main__":
    split()
