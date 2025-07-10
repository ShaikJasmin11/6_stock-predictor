# main.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from src.preprocess import load_stock_data, prepare_data

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_all_stocks():
    df = load_stock_data()
    os.makedirs("models", exist_ok=True)

    for stock in df.columns:
        print(f"📈 Training model for {stock}...")
        scaler, X, y = prepare_data(df[stock].values)

        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = build_model((X.shape[1], 1))
        model.fit(X, y, epochs=10, batch_size=32, verbose=0,
                  callbacks=[EarlyStopping(patience=2)])

        model.save(f"models/{stock}_model.h5")
        np.save(f"models/{stock}_scaler.npy", scaler.scale_)
        np.save(f"models/{stock}_min.npy", scaler.min_)

        print(f"✅ Saved model for {stock}")

if __name__ == '__main__':
    train_all_stocks()
