import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

def get_stock_data(ticker, start='2022-01-01', end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end)
    return data

def prepare_data(data, days_ahead=1):
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Target'] = data['Close'].shift(-days_ahead)
    data = data.dropna()
    return data

def train_model(data):
    X = data[['MA10', 'MA50']]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def predict_future_price(model, recent_data, days_ahead):
    ma10 = recent_data['Close'][-10:].mean()
    ma50 = recent_data['Close'][-50:].mean()
    features = np.array([ma10, ma50]).reshape(1, -1)
    prediction = model.predict(features)
    return round(prediction[0], 2)
