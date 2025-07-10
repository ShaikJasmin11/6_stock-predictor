#  Stock Price Predictor (LSTM)

> RISE Internship Project 6 – Tamizhan Skills  
> Built with TensorFlow, LSTM, NumPy, and Streamlit

A deep learning-based web app that predicts the next-day closing price of selected stocks using historical trends. This is the sixth project from the **Machine Learning & AI** track of the RISE Internship by Tamizhan Skills.

---

##  Project Objective

To build a stock price forecasting system that:
  - Loads multi-stock time-series data from a CSV file
  - Scales and reshapes data into LSTM-friendly format
  - Trains an **LSTM neural network** per stock
  - Predicts the **next day’s price** using the past 60 days
  - Offers a simple **Streamlit UI** to select and forecast

---

##  Tech Stack

- **Python**
- **Pandas / NumPy**
- **TensorFlow (LSTM Model)**
- **Scikit-learn (MinMaxScaler)**
- **Streamlit** (for real-time frontend UI)

---

##  Project Structure

```bash
stock-price-predictor/
├── app.py                     # Streamlit frontend for real-time price prediction
├── main.py                    # LSTM training loop for all stocks
├── requirements.txt           # All required packages
├── data/
│   └── stock_data.csv         # CSV with 5 stock columns: Stock_1 to Stock_5
├── models/
│   ├── Stock_X_model.h5       # Trained LSTM model for each stock
│   ├── Stock_X_scaler.npy     # Scaler parameters for each stock
│   └── Stock_X_min.npy        # Minimum value for inverse scaling
├── src/
│   └── preprocess.py          # Data loading and preparation logic
└── README.md                  # You’re reading it now 😉
```

---

## Dataset

- Source: Stock Price Prediction Dataset – Kaggle(https://www.kaggle.com/datasets/mrsimple07/stock-price-prediction)
- File: stock_data.csv
- Format:
   - Columns: empty, Stock_1, Stock_2, Stock_3, Stock_4, Stock_5
   - Each column represents historical daily closing prices for a stock
- Preprocessed using 60-day lookback windows for LSTM training

---

## How to Run

- Step 1: Install Dependencies
  
```bash
  pip install -r requirements.txt
```

- Step 2: Train the Model
  
```bash
  python main.py
```

- Step 3: Launch the Web App
  
```bash
  streamlit run app.py
```

  ---

## Model Performance

✅ Predicts next-day stock prices using LSTM
✅ Trained separately for each stock using 60-day historical windows
✅ Scaled using MinMaxScaler for smoother convergence
✅ Real-time forecast output from the Streamlit UI

---

## Highlights

- Uses LSTM, ideal for time-series sequence learning
- Minimal data preprocessing via MinMaxScaler
- Saves model and scaling params separately for reuse
- Streamlit app to select any of 5 stocks and get instant prediction
- Lightweight and modular codebase — easy to upgrade with graphs or APIs
  
---

## Acknowledgements

Thanks to Tamizhan Skills for the RISE Internship opportunity.

Inspired by real-world financial forecasting use cases in fintech, trading, and AI analytics.

Built by @ShaikJasmin11
