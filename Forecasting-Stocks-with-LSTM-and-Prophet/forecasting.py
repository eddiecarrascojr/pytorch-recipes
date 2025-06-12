import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import matplotlib.pyplot as plt

# --- 1. Data Fetching and Preparation ---
# Fetch historical stock data for a specific ticker
ticker = 'AAPL' # Example: Apple Inc.
data = yf.download(ticker, start='2020-01-01', end='2024-12-31')

# We'll use the 'Close' price for forecasting
data = data[['Close']]
data = data.reset_index() # Reset index to get 'Date' as a column

# --- 2. PyTorch LSTM Model ---

# A. Data Preprocessing for LSTM
# Normalize the data to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# Split data into training and testing sets
training_size = int(len(data_scaled) * 0.80)
test_size = len(data_scaled) - training_size
train_data, test_data = data_scaled[0:training_size,:], data_scaled[training_size:len(data_scaled),:1]

# Function to create sequences for the LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Reshape into X=t,t+1,t+2,...t+99 and Y=t+100
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)

# B. LSTM Model Definition
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# C. Training the LSTM Model
model = StockLSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for i in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = loss_function(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (i+1) % 10 == 0:
      print(f'LSTM Epoch {i+1}, Loss: {loss.item()}')

# D. LSTM Prediction
model.eval()
with torch.no_grad():
    train_predict = model(X_train_tensor).numpy()
    test_predict = model(X_test_tensor).numpy()

# Inverse transform to get actual prices
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_orig = scaler.inverse_transform(y_train_tensor.numpy())
y_test_orig = scaler.inverse_transform(y_test_tensor.numpy())


# --- 3. Facebook Prophet Model ---

# A. Data Preparation for Prophet
# Prophet requires columns 'ds' (datestamp) and 'y' (value)
prophet_data = data.rename(columns={'Date': 'ds', 'Close': 'y'})
prophet_train = prophet_data.iloc[:training_size]
prophet_test = prophet_data.iloc[training_size:]

# B. Fitting Prophet Model
prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(prophet_train)

# C. Prophet Prediction
future = prophet_model.make_future_dataframe(periods=len(prophet_test))
forecast = prophet_model.predict(future)
prophet_predictions = forecast['yhat'][-len(prophet_test):]


# --- 4. Model Comparison & Visualization ---

# A. Performance Metrics
print("\n--- Model Performance ---")
# LSTM Metrics
lstm_train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_predict))
lstm_test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_predict))
print(f"LSTM Train RMSE: {lstm_train_rmse:.2f}")
print(f"LSTM Test RMSE:  {lstm_test_rmse:.2f}")

# Prophet Metrics
prophet_test_rmse = np.sqrt(mean_squared_error(prophet_test['y'], prophet_predictions))
print(f"Prophet Test RMSE: {prophet_test_rmse:.2f}")
print("-------------------------\n")


# B. Visualization
plt.figure(figsize=(16,8))
plt.title(f'Stock Price Prediction for {ticker}')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)

# Plot actual prices
plt.plot(data['Date'], data['Close'], label='Actual Price')

# Plot LSTM Predictions
lstm_plot_data = np.empty_like(data_scaled)
lstm_plot_data[:, :] = np.nan
lstm_plot_data[time_step:len(train_predict)+time_step, :] = scaler.inverse_transform(train_data[time_step:])
lstm_plot_data[len(train_predict)+(time_step*2)+1:len(data_scaled)-1, :] = test_predict
plt.plot(data['Date'], lstm_plot_data, 'r', label='LSTM Prediction')


# Plot Prophet Predictions
plt.plot(prophet_test['ds'], prophet_predictions, 'g', label='Prophet Prediction')

plt.legend(loc='lower right')
plt.grid(True)
plt.show()

