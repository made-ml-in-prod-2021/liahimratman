import requests
import pandas as pd


DATA_PATH = "ml_project/data/heart.csv"


if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={
            "ids": list(range(data.shape[0])),
            "features": data.values.tolist(),
            "columns": data.columns.tolist()
        },
    )
    print(response.status_code)
    print(response.json())
