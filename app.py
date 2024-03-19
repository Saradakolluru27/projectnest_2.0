import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os  # Import the os module to set environment variables

# Set environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the Keras model
model_path = r'C:\Users\sarada\Downloads\Stock_Market_Prediction_ML_extract\Stock Predictions Model.keras'  # Update this path to point to your saved model file with .h5 extension
model = load_model(model_path)

# Header
st.header('Stock Market Predictor')

# Input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GC=F')

# Download stock data
start = '2014-01-01'
end = '2023-12-31'
data = yf.download(stock, start, end)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Preprocess the data
data_train = pd.DataFrame(data.Close[:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plot Price vs MA50
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(ma_50_days, 'r', label='MA50')
ax1.plot(data.Close, 'g', label='Price')
ax1.legend()
st.pyplot(fig1)

# Plot Price vs MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(ma_50_days, 'r', label='MA50')
ax2.plot(ma_100_days, 'b', label='MA100')
ax2.plot(data.Close, 'g', label='Price')
ax2.legend()
st.pyplot(fig2)

# Plot Price vs MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(ma_100_days, 'r', label='MA100')
ax3.plot(ma_200_days, 'b', label='MA200')
ax3.plot(data.Close, 'g', label='Price')
ax3.legend()
st.pyplot(fig3)

# Prepare data for prediction
x = []
y_true = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y_true.append(data_test_scale[i, 0])

x, y_true = np.array(x), np.array(y_true)

# Make predictions
predictions = model.predict(x)

# Rescale predictions and true values
predictions = predictions * scaler.scale_
y_true = y_true * scaler.scale_

# Plot Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(y_true, 'r', label='Original Price')
ax4.plot(predictions, 'g', label='Predicted Price')
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
ax4.legend()
st.pyplot(fig4)
