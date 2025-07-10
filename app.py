# app.py

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from src.preprocess import load_stock_data

st.set_page_config(page_title="📊 Stock Predictor", layout="centered")
st.title("📊 Stock Price Predictor (LSTM)")
st.write("Predict the next day’s price of a stock using deep learning.")

df = load_stock_data()
stock_options = df.columns.tolist()
selected_stock = st.selectbox("Select a stock", stock_options)

if st.button("Predict Next Day Price"):
    model = load_model(f"models/{selected_stock}_model.h5")
    series = df[selected_stock].values.reshape(-1, 1)[-60:]

    # Load scaler info
    scale = np.load(f"models/{selected_stock}_scaler.npy")
    min_ = np.load(f"models/{selected_stock}_min.npy")

    scaled_input = (series - min_) / scale
    X_pred = scaled_input.reshape(1, 60, 1)
    prediction = model.predict(X_pred)[0][0]

    # Inverse scale
    predicted_price = (prediction * scale[0]) + min_[0]
    st.success(f"📈 Predicted Next Day Price: ₹{predicted_price:.2f}")
