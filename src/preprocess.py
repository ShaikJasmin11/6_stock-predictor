# src/preprocess.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_stock_data(path='data/stock_data.csv'):
    df = pd.read_csv(path)
    df.drop(columns=df.columns[0], inplace=True)  # Drop the unnamed index column
    return df

def prepare_data(series, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])

    return (
        scaler,
        X, y
    )
