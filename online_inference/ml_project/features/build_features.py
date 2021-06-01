import pandas as pd


def make_features(transformers: dict, data: pd.DataFrame, mode="val") -> (pd.DataFrame, dict):
    """
    Make features
    :param transformers: Dict with column transformer and custom standard scaler transformer
    :param data: input data
    :param mode: processing mode
    :return: pd.DataFrame (preprocessed data)
    """
    if mode == "train":
        transformers["column_transformer"].fit(data)

    data = pd.DataFrame(transformers["column_transformer"].transform(data))

    if mode == "train":
        transformers["standard_scaler_transformer"].fit(data)

    return pd.DataFrame(transformers["standard_scaler_transformer"].transform(data)), transformers
