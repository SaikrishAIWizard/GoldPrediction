from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .forms import SignUpForm
from django.contrib.auth.forms import AuthenticationForm

from polygon import RESTClient

from datetime import timedelta
import datetime
import pytz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import pickle
import json

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pmdarima import auto_arima

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.losses as losses



def home(request):
    return render(request, 'home.html')


def get_candle_data(stock_name,multiplier_val,timespan_val, from_val , to_val):
    polygonAPIkey = "XQUrVaX7b9lIYNiCiIvvz6GSVe05zw_e"
    # Create client and authentication
    broker = RESTClient(polygonAPIkey)
    data = broker.get_aggs(ticker=stock_name,
                           multiplier=multiplier_val,
                           timespan=timespan_val,
                           from_=from_val,
                           to=to_val)
    return data

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

# Function to compute SHA256 hash of a file
def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


# Define the CNN-Bidirectional LSTM model creation function
def create_cnn_bi_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)

    x = Bidirectional(LSTM(units=50, return_sequences=False))(x)
    x = Dropout(0.2)(x)

    x = Dense(units=50, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(units=1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_sequences(data, seq_length=60):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:i+seq_length])
                    y.append(data[i+seq_length])
                return np.array(X), np.array(y)

def create_xsequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i+seq_length])
    return np.array(X)


def train_model(request):
    df = pd.read_csv(r"firstapp\Models\final_gold_data.csv")
    last_updated = df['date'].iloc[-1]
    if request.method == "POST":
        #model_name = request.POST["model_name"]
        #train_date = request.POST["train_date"]
        #print(model_name,train_date)

        # Load the new data CSV path
        new_data_path = r'firstapp\Models\final_gold_data.csv'

        df = pd.read_csv(new_data_path, parse_dates=['date'], index_col='date')

        # Get today's date
        today = datetime.datetime.now()
        est_timezone = pytz.timezone('Asia/Kolkata')
        est_dt = today.astimezone(est_timezone)
        minutes_value = est_dt.minute
        flor_val = minutes_value//15
        new_dt = est_dt.replace(minute=0)
        new_dt = new_dt + datetime.timedelta(minutes=15*flor_val)
        today = new_dt.replace(second=0, microsecond=0)
        new_dt = today.replace(hour=0, minute=0, second=0, microsecond=0)

        stock_name = "C:XAUUSD"
        data = get_candle_data(stock_name=stock_name, multiplier_val=1, timespan_val="day",
                                from_val=new_dt - timedelta(days=1000),
                                to_val=today)
        data = pd.DataFrame(data)
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms').dt.date
        data = data.set_index('date')

        df = df.reset_index()
        data = data.reset_index()

        # Ensure 'date' column is datetime
        df['date'] = pd.to_datetime(df['date'])
        data['date'] = pd.to_datetime(data['date'])

        # Step 2: Select only required columns from each DataFrame
        df1_small = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df2_small = data[['date', 'open', 'high', 'low', 'close', 'volume']]

        # Get the max date in df1
        max_date_df1 = df1_small['date'].max()
        # Select rows in df2 where date is strictly greater than max date in df1
        df2_new_after_df1 = df2_small[df2_small['date'] > max_date_df1]
        df2_new_after_df1.to_csv(r'firstapp\Models\Newlyadded_gold_data.csv', index=False)

        combined_df = pd.merge(df1_small, df2_small, on='date', how='outer', suffixes=('_df1', '_df2'))

        # Choose df1 columns first, fallback to df2 if missing
        for col in ['open', 'high', 'low', 'close', 'volume']:
            combined_df[col] = combined_df[f'{col}_df1'].combine_first(combined_df[f'{col}_df2'])

        # Keep final columns
        final_df = combined_df[['date', 'open', 'high', 'low', 'close', 'volume']]

        # Sort by date ascending and reset index
        final_df = final_df.sort_values('date').reset_index(drop=True)
        final_df.to_csv(new_data_path, index=False)

        # Compute hash of new data
        new_data_hash = get_file_hash(new_data_path)

        # File storing hashes of datasets already used for training
        hash_file = r'firstapp\Models\trained_data_hashes.txt'

        # Load known hashes or initialize empty list
        try:
            with open(hash_file, 'r') as f:
                known_hashes = f.read().splitlines()
        except FileNotFoundError:
            known_hashes = []

        All_model_metrics = {}
        # Check if new data was already used for retraining
        if new_data_hash in known_hashes:
            print("Data already used for training. Skipping retraining.")
        else:
            print("New data detected. Starting retraining.")

            # Load your data (assuming datetime index and 'Close' column)
            df = pd.read_csv(new_data_path,parse_dates=['date'], index_col='date')
            close_prices = df['close']

            # Assume `series` is your time series data
            stepwise_fit = auto_arima(close_prices, start_p=0, start_q=0,
                          max_p=10, max_q=10,
                          d=None,           # let auto find d using tests
                          seasonal=False,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
                        
            # Fit ARIMA model (adjust order as needed)
            arima_order = stepwise_fit.order
            #arima_order = (7,1,1)
            arima_model = ARIMA(close_prices, order=arima_order)
            arima_fit = arima_model.fit()

            # Get ARIMA predictions
            arima_pred = arima_fit.predict(start=arima_fit.loglikelihood_burn, end=len(close_prices)-1)

            # Calculate residuals (nonlinear components)
            residuals = close_prices[arima_fit.loglikelihood_burn:] - arima_pred
            residuals = residuals.dropna()

            # Scale residuals
            scaler = MinMaxScaler()
            residuals_scaled = scaler.fit_transform(residuals.values.reshape(-1,1))

            seq_length = 60
            
            X, y = create_sequences(residuals_scaled, seq_length)

            # Split train/test sets (80% train, 20% test)
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Reshape for LSTM input
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            input_shape=(seq_length,1)
            model = create_cnn_bi_lstm_model(input_shape)
            history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

            # Predict residuals
            lstm_pred_scaled = model.predict(X_test)
            lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
            # Align ARIMA predictions for test period
            arima_pred_test = arima_pred[-len(lstm_pred):]
            # Final hybrid prediction = ARIMA + LSTM residual correction
            hybrid_pred = arima_pred_test + lstm_pred
            # Actual values for test period
            actual_test = close_prices[-len(lstm_pred):]
   
            mae = mean_absolute_error(actual_test, hybrid_pred)
            rmse = np.sqrt(mean_squared_error(actual_test, hybrid_pred))
            mape = np.mean(np.abs((actual_test - hybrid_pred) / actual_test)) * 100
            r2_lstm = r2_score(actual_test, hybrid_pred) * 100

            All_model_metrics['LSTM_daymodel'] = {'mae':mae,'rmse':rmse,'mape':mape,'r2_lstm':r2_lstm}

            model.save(r'firstapp\Models\gold\gold_lstm_model_day.h5')

            with open(r'firstapp\Models\gold\gold_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            # Assume `history` = model.fit(...) returned history object
            with open(r'firstapp\Models\gold\gold_history_day.pkl', 'wb') as f:
                pickle.dump(history, f)

            # Save the ARIMA model
            with open(r'firstapp\Models\gold\gold_arima_day.pkl', 'wb') as f:
                pickle.dump(arima_fit, f)


            #Train the Week Wise Predicted Model
            # Define aggregation for weekly resampling
            week_agg = {
                'open': 'first',    # Open price of the first trading day in the week
                'high': 'max',      # Highest price in the week
                'low': 'min',       # Lowest price in the week
                'close': 'last',    # Close price of the last trading day in the week
                'volume': 'sum'     # Total volume traded in the week
            }
            df_weekly = df.resample('W-FRI').agg(week_agg).dropna().reset_index()
            close_prices = df_weekly['close']

            stepwise_fit = auto_arima(close_prices, start_p=0, start_q=0,
                          max_p=10, max_q=10,
                          d=None,           # let auto find d using tests
                          seasonal=False,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
                        
            # Fit ARIMA model (adjust order as needed)
            arima_order = stepwise_fit.order
            #arima_order = (7,1,1)
            arima_model = ARIMA(close_prices, order=arima_order)
            arima_fit = arima_model.fit()

            # Get ARIMA predictions
            arima_pred = arima_fit.predict(start=arima_fit.loglikelihood_burn, end=len(close_prices)-1)

            # Calculate residuals (nonlinear components)
            residuals = close_prices[arima_fit.loglikelihood_burn:] - arima_pred
            residuals = residuals.dropna()

            # Scale residuals
            scaler = MinMaxScaler()
            residuals_scaled = scaler.fit_transform(residuals.values.reshape(-1,1))

            seq_length = 30
            X, y = create_sequences(residuals_scaled, seq_length)

            # Split train/test sets (80% train, 20% test)
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Reshape for LSTM input
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            input_shape=(seq_length,1)
            model = create_cnn_bi_lstm_model(input_shape)
            history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

            # Predict residuals
            lstm_pred_scaled = model.predict(X_test)
            lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
            # Align ARIMA predictions for test period
            arima_pred_test = arima_pred[-len(lstm_pred):]
            # Final hybrid prediction = ARIMA + LSTM residual correction
            hybrid_pred = arima_pred_test + lstm_pred
            # Actual values for test period
            actual_test = close_prices[-len(lstm_pred):]
   
            mae = mean_absolute_error(actual_test, hybrid_pred)
            rmse = np.sqrt(mean_squared_error(actual_test, hybrid_pred))
            mape = np.mean(np.abs((actual_test - hybrid_pred) / actual_test)) * 100
            r2_lstm = r2_score(actual_test, hybrid_pred) * 100
            
            All_model_metrics['LSTM_weekmodel'] = {'mae':mae,'rmse':rmse,'mape':mape,'r2_lstm':r2_lstm}


            model.save(r'firstapp\Models\gold\gold_lstm_model_week.h5')

            with open(r'firstapp\Models\gold\gold_scaler_week.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            # Assume `history` = model.fit(...) returned history object
            with open(r'firstapp\Models\gold\gold_history_week.pkl', 'wb') as f:
                pickle.dump(history, f)

            # Save the ARIMA model
            with open(r'firstapp\Models\gold\gold_arima_week.pkl', 'wb') as f:
                pickle.dump(arima_fit, f)


            #Train the Month Wise Predicted Model
            month_agg = {
                'open': 'first',    # Open price of the first trading day in the month
                'high': 'max',      # Highest price in the month
                'low': 'min',       # Lowest price in the month
                'close': 'last',    # Close price of the last trading day in the month
                'volume': 'sum'     # Total volume traded in the month
            }

            df_monthly = df.resample('M').agg(month_agg).dropna().reset_index()
            close_prices = df_monthly['close']

            stepwise_fit = auto_arima(close_prices, start_p=0, start_q=0,
                          max_p=10, max_q=10,
                          d=None,           # let auto find d using tests
                          seasonal=False,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
                        
            # Fit ARIMA model (adjust order as needed)
            arima_order = stepwise_fit.order
            arima_model = ARIMA(close_prices, order=arima_order)
            arima_fit = arima_model.fit()

            # Get ARIMA predictions
            arima_pred = arima_fit.predict(start=arima_fit.loglikelihood_burn, end=len(close_prices)-1)

            # Calculate residuals (nonlinear components)
            residuals = close_prices[arima_fit.loglikelihood_burn:] - arima_pred
            residuals = residuals.dropna()

            # Scale residuals
            scaler = MinMaxScaler()
            residuals_scaled = scaler.fit_transform(residuals.values.reshape(-1,1))

            seq_length = 24
            X, y = create_sequences(residuals_scaled, seq_length)

            # Split train/test sets (80% train, 20% test)
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Reshape for LSTM input
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            input_shape=(seq_length,1)
            model = create_cnn_bi_lstm_model(input_shape)
            history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

            # Predict residuals
            lstm_pred_scaled = model.predict(X_test)
            lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
            # Align ARIMA predictions for test period
            arima_pred_test = arima_pred[-len(lstm_pred):]
            # Final hybrid prediction = ARIMA + LSTM residual correction
            hybrid_pred = arima_pred_test + lstm_pred
            # Actual values for test period
            actual_test = close_prices[-len(lstm_pred):]
   
            mae = mean_absolute_error(actual_test, hybrid_pred)
            rmse = np.sqrt(mean_squared_error(actual_test, hybrid_pred))
            mape = np.mean(np.abs((actual_test - hybrid_pred) / actual_test)) * 100
            r2_lstm = r2_score(actual_test, hybrid_pred) * 100
            
            All_model_metrics['LSTM_monthmodel'] = {'mae':mae,'rmse':rmse,'mape':mape,'r2_lstm':r2_lstm}


            model.save(r'firstapp\Models\gold\gold_lstm_model_month.h5')

            with open(r'firstapp\Models\gold\gold_scaler_month.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            # Assume `history` = model.fit(...) returned history object
            with open(r'firstapp\Models\gold\gold_history_month.pkl', 'wb') as f:
                pickle.dump(history, f)

            # Save the ARIMA model
            with open(r'firstapp\Models\gold\gold_arima_month.pkl', 'wb') as f:
                pickle.dump(arima_fit, f)

            #Train the Year Wise Predicted Model
            # Define aggregation for yearly resampling
            year_agg = {
                'open': 'first',    # Open price of the first trading day in the year
                'high': 'max',      # Highest price in the year
                'low': 'min',       # Lowest price in the year
                'close': 'last',    # Close price of the last trading day in the year
                'volume': 'sum'     # Total volume traded in the year
            }
            df_yearly = df.resample('Y').agg(year_agg).dropna().reset_index()

            close_prices = df_yearly['close']
            stepwise_fit = auto_arima(close_prices, start_p=0, start_q=0,
                          max_p=10, max_q=10,
                          d=None,           # let auto find d using tests
                          seasonal=False,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
                        
            # Fit ARIMA model (adjust order as needed)
            arima_order = stepwise_fit.order
            arima_model = ARIMA(close_prices, order=arima_order)
            arima_fit = arima_model.fit()

            # Get ARIMA predictions
            arima_pred = arima_fit.predict(start=arima_fit.loglikelihood_burn, end=len(close_prices)-1)

            # Calculate residuals (nonlinear components)
            residuals = close_prices[arima_fit.loglikelihood_burn:] - arima_pred
            residuals = residuals.dropna()

            # Scale residuals
            scaler = MinMaxScaler()
            residuals_scaled = scaler.fit_transform(residuals.values.reshape(-1,1))

            seq_length = 3
            X, y = create_sequences(residuals_scaled, seq_length)

            # Split train/test sets (80% train, 20% test)
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Reshape for LSTM input
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            input_shape=(seq_length,1)
            model = create_cnn_bi_lstm_model(input_shape)
            history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

            # Predict residuals
            lstm_pred_scaled = model.predict(X_test)
            lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
            # Align ARIMA predictions for test period
            arima_pred_test = arima_pred[-len(lstm_pred):]
            # Final hybrid prediction = ARIMA + LSTM residual correction
            hybrid_pred = arima_pred_test + lstm_pred
            # Actual values for test period
            actual_test = close_prices[-len(lstm_pred):]
   
            mae = mean_absolute_error(actual_test, hybrid_pred)
            rmse = np.sqrt(mean_squared_error(actual_test, hybrid_pred))
            mape = np.mean(np.abs((actual_test - hybrid_pred) / actual_test)) * 100
            r2_lstm = r2_score(actual_test, hybrid_pred) * 100
            
            All_model_metrics['LSTM_yearmodel'] = {'mae':mae,'rmse':rmse,'mape':mape,'r2_lstm':r2_lstm}


            model.save(r'firstapp\Models\gold\gold_lstm_model_year.h5')

            with open(r'firstapp\Models\gold\gold_scaler_year.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            # Assume `history` = model.fit(...) returned history object
            with open(r'firstapp\Models\gold\gold_history_year.pkl', 'wb') as f:
                pickle.dump(history, f)

            # Save the ARIMA model
            with open(r'firstapp\Models\gold\gold_arima_year.pkl', 'wb') as f:
                pickle.dump(arima_fit, f)

            with open(r'firstapp\Models\gold\All_model_metrics.json', 'w') as f:
                json.dump(All_model_metrics, f)

            # Step 11: Update known hashes file to include new data hash to prevent re-retraining on same data
            known_hashes.append(new_data_hash)
            with open(hash_file, 'w') as f:
                f.write('\n'.join(known_hashes))
        
            print("All_model_metrics",All_model_metrics)
        context = {'last_updated_date': last_updated,'all_model_metrics_json': json.dumps(All_model_metrics)}
        return render(request, 'firstapp/train_model.html',context)

        # Redirect or render a results page as needed
    else:   
        return render(request, 'firstapp/train_model.html', {
        'last_updated_date': last_updated
                                            })

def forecast(request):
    if request.method == 'POST':
        days_forecast = int(request.POST.get('days_forecast', 0))
        weeks_forecast = int(request.POST.get('weeks_forecast', 0))
        months_forecast = int(request.POST.get('months_forecast', 0))
        years_forecast = int(request.POST.get('years_forecast', 0))

        stock_name = "C:XAUUSD"
        # Day   
        model_day = load_model(r'firstapp\Models\gold\gold_lstm_model_day.h5', custom_objects={'mse': losses.MeanSquaredError()})

        with open(r'firstapp\Models\gold\gold_scaler.pkl', 'rb') as f:
            scaler_day = pickle.load(f)

        with open(r'firstapp\Models\gold\gold_history_day.pkl', 'rb') as f:
            history_day = pickle.load(f)

        with open(r'firstapp\Models\gold\gold_arima_day.pkl', 'rb') as f:
            arima_fit_day = pickle.load(f)

        #Week
        model_week = load_model(r'firstapp\Models\gold\gold_lstm_model_week.h5', custom_objects={'mse': losses.MeanSquaredError()})

        with open(r'firstapp\Models\gold\gold_scaler_week.pkl', 'rb') as f:
            scaler_week = pickle.load(f)

        with open(r'firstapp\Models\gold\gold_history_week.pkl', 'rb') as f:
            history_week = pickle.load(f)

        with open(r'firstapp\Models\gold\gold_arima_week.pkl', 'rb') as f:
            arima_fit_week = pickle.load(f)

        #Month
        model_month = load_model(r'firstapp\Models\gold\gold_lstm_model_month.h5', custom_objects={'mse': losses.MeanSquaredError()})

        with open(r'firstapp\Models\gold\gold_scaler_month.pkl', 'rb') as f:
            scaler_month = pickle.load(f)

        with open(r'firstapp\Models\gold\gold_history_month.pkl', 'rb') as f:
            history_month = pickle.load(f)

        with open(r'firstapp\Models\gold\gold_arima_month.pkl', 'rb') as f:
            arima_fit_month = pickle.load(f)

        #Year
        model_year = load_model(r'firstapp\Models\gold\gold_lstm_model_year.h5', custom_objects={'mse': losses.MeanSquaredError()})

        with open(r'firstapp\Models\gold\gold_scaler_year.pkl', 'rb') as f:
            scaler_year = pickle.load(f)

        with open(r'firstapp\Models\gold\gold_history_year.pkl', 'rb') as f:
            history_year = pickle.load(f)

        with open(r'firstapp\Models\gold\gold_arima_year.pkl', 'rb') as f:
            arima_fit_year = pickle.load(f)

        
        # Get today's date
        today = datetime.datetime.now()

        # Convert to Eastern Standard Time (EST)
        est_timezone = pytz.timezone('Asia/Kolkata')
        est_dt = today.astimezone(est_timezone)
        minutes_value = est_dt.minute
        flor_val = minutes_value//15
        new_dt = est_dt.replace(minute=0)
        new_dt = new_dt + datetime.timedelta(minutes=15*flor_val)
        today = new_dt.replace(second=0, microsecond=0)
        new_dt = today.replace(hour=0, minute=0, second=0, microsecond=0)

        #Day wise Prediction

        data = get_candle_data(stock_name=stock_name, multiplier_val=1, timespan_val="day",
                                from_val=new_dt - timedelta(days=200),
                                to_val=today)
        data = pd.DataFrame(data)
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms').dt.date

        data = data.set_index('date')
        df_day = data.tail(60).copy()
        close_prices = df_day['close'].values.reshape(-1, 1)
        scaled_close = scaler_day.fit_transform(close_prices)
        seq_length = 60
    
        # For prediction, we use the last 60 days as input sequence of shape (1, 60, 1)
        X_input = create_xsequences(scaled_close, seq_length)[-1:]

        predictions_scaled = []

        current_input = X_input  # shape (1, 60, 1)

        forecast_next_days = days_forecast

        for _ in range(forecast_next_days):
            pred_scaled = model_day.predict(current_input)[0, 0]
            predictions_scaled.append(pred_scaled)
            # Update the input by adding prediction and removing oldest
            current_input = np.append(current_input[:,1:,:], [[[pred_scaled]]], axis=1)

        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        predictions_day = scaler_day.inverse_transform(predictions_scaled).flatten()

        print("Next {} days predicted close prices:".format(forecast_next_days))

        close_prices_flat = close_prices.flatten()
        combined_day = np.concatenate([close_prices_flat, predictions_day])
        arima_pred = arima_fit_day.predict(start=arima_fit_day.loglikelihood_burn, end=len(combined_day)-1)
        day_values_list = (arima_pred[-forecast_next_days:] + predictions_day).values

        # Specify start date and frequency
        start_date = str(df_day.tail(1).index[0])
        #start_date = '2025-09-12'  # for example, tomorrow's date
        day_dates_list = pd.date_range(start=start_date, periods=len(day_values_list), freq='D')  # 'D' for daily
        print("dates",day_dates_list)
        # Convert DatetimeIndex to list of ISO formatted strings
        day_dates_list = day_dates_list.strftime('%Y-%m-%d').tolist()


        
        #Week wise Prediction

        data = get_candle_data(stock_name=stock_name, multiplier_val=1, timespan_val="week",
                                from_val=new_dt - timedelta(days=700),
                                to_val=today)
        data = pd.DataFrame(data)
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms').dt.date

        data = data.set_index('date')
        df_week = data.tail(30).copy()
        close_prices = df_week['close'].values.reshape(-1, 1)
        scaled_close = scaler_week.fit_transform(close_prices)
        seq_length = 30
    
        # For prediction, we use the last 60 days as input sequence of shape (1, 60, 1)
        X_input = create_xsequences(scaled_close, seq_length)[-1:]

        predictions_scaled = []

        current_input = X_input  # shape (1, 60, 1)

        forecast_next_days = weeks_forecast

        for _ in range(forecast_next_days):
            pred_scaled = model_week.predict(current_input)[0, 0]
            predictions_scaled.append(pred_scaled)
            # Update the input by adding prediction and removing oldest
            current_input = np.append(current_input[:,1:,:], [[[pred_scaled]]], axis=1)

        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        predictions_week = scaler_week.inverse_transform(predictions_scaled).flatten()

        print("Next {} weeks predicted close prices:".format(forecast_next_days))

        close_prices_flat = close_prices.flatten()
        combined_week = np.concatenate([close_prices_flat, predictions_week])
        arima_pred = arima_fit_week.predict(start=arima_fit_week.loglikelihood_burn, end=len(combined_week)-1)
        week_values_list = (arima_pred[-forecast_next_days:] + predictions_week).values
        print(week_values_list)

        start_date = str(df_week.tail(1).index[0])  # Assuming df_week is your time series DataFrame for weeks
        week_dates_list = pd.date_range(start=start_date, periods=len(week_values_list), freq='W')  # 'W' for weekly
        print("dates_week",week_dates_list)

        week_dates_list = week_dates_list.strftime('%Y-%m-%d').tolist()

        
        #Month wise Prediction

        data = get_candle_data(stock_name=stock_name, multiplier_val=1, timespan_val="month",
                                from_val=new_dt - timedelta(days=2000),
                                to_val=today)
        data = pd.DataFrame(data)
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms').dt.date

        data = data.set_index('date')
        df_month = data.tail(24).copy()
        close_prices = df_month['close'].values.reshape(-1, 1)
        scaled_close = scaler_month.fit_transform(close_prices)
        seq_length = 24
    
        # For prediction, we use the last 60 days as input sequence of shape (1, 60, 1)
        X_input = create_xsequences(scaled_close, seq_length)[-1:]

        predictions_scaled = []

        current_input = X_input  # shape (1, 60, 1)

        forecast_next_days = months_forecast

        for _ in range(forecast_next_days):
            pred_scaled = model_month.predict(current_input)[0, 0]
            predictions_scaled.append(pred_scaled)
            # Update the input by adding prediction and removing oldest
            current_input = np.append(current_input[:,1:,:], [[[pred_scaled]]], axis=1)

        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        predictions_month = scaler_month.inverse_transform(predictions_scaled).flatten()

        print("Next {} Months predicted close prices:".format(forecast_next_days))

        close_prices_flat = close_prices.flatten()
        combined_month = np.concatenate([close_prices_flat, predictions_month])
        arima_pred = arima_fit_month.predict(start=arima_fit_month.loglikelihood_burn, end=len(combined_month)-1)
        month_values_list = (arima_pred[-forecast_next_days:] + predictions_month).values
        print(month_values_list)

        start_date = str(df_month.tail(1).index[0])  # Assuming df_week is your time series DataFrame for weeks
        month_dates_list = pd.date_range(start=start_date, periods=len(month_values_list), freq='m')  # 'W' for weekly
        print("month_dates_list",month_dates_list)

        month_dates_list = month_dates_list.strftime('%Y-%m-%d').tolist()

        #Year wise Prediction

        data = get_candle_data(stock_name=stock_name, multiplier_val=1, timespan_val="year",
                                from_val=new_dt - timedelta(days=10000),
                                to_val=today)
        data = pd.DataFrame(data)
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms').dt.date

        data = data.set_index('date')
        print("data is",data)
        df_month = data.tail(3).copy()
        close_prices = df_month['close'].values.reshape(-1, 1)
        scaled_close = scaler_year.fit_transform(close_prices)
        seq_length = 3
    
        # For prediction, we use the last 60 days as input sequence of shape (1, 60, 1)
        X_input = create_xsequences(scaled_close, seq_length)[-1:]

        predictions_scaled = []

        current_input = X_input  # shape (1, 60, 1)
        print("current_input is:",current_input)
        forecast_next_days = years_forecast

        for _ in range(forecast_next_days):
            pred_scaled = model_year.predict(current_input)[0, 0]
            predictions_scaled.append(pred_scaled)
            # Update the input by adding prediction and removing oldest
            current_input = np.append(current_input[:,1:,:], [[[pred_scaled]]], axis=1)

        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        predictions_month = scaler_year.inverse_transform(predictions_scaled).flatten()

        print("Next {} Months predicted close prices:".format(forecast_next_days))

        close_prices_flat = close_prices.flatten()
        combined_month = np.concatenate([close_prices_flat, predictions_month])
        arima_pred = arima_fit_year.predict(start=arima_fit_year.loglikelihood_burn, end=len(combined_month)-1)
        year_values_list = (arima_pred[-forecast_next_days:] + predictions_month).values
        print(year_values_list)

        start_date = str(df_month.tail(1).index[0])  # Assuming df_week is your time series DataFrame for weeks
        year_dates_list = pd.date_range(start=start_date, periods=len(year_values_list), freq='y')  # 'W' for weekly
        print("year_dates_list",year_dates_list)

        year_dates_list = year_dates_list.strftime('%Y-%m-%d').tolist()

        context = {
                'day_dates': json.dumps(day_dates_list),     # e.g., ['2025-09-12', '2025-09-13', ...]
                'day_values': json.dumps(day_values_list.tolist()),   # e.g., [331.07, 331.19, ...]
                'week_dates': json.dumps(week_dates_list),
                'week_values': json.dumps(week_values_list.tolist()),
                'month_dates': json.dumps(month_dates_list),
                'month_values': json.dumps(month_values_list.tolist()),
                # pass other form values if needed
                'year_labels': json.dumps(year_dates_list),
                'year_yhat': json.dumps(year_values_list.tolist()),
                        }
        return render(request, 'firstapp/forecast.html', context)
    else:
        return render(request, 'firstapp/forecast.html')



def model_graphs(request):
    file_path = r"firstapp\Models\gold\All_model_metrics.json"
    with open(file_path, 'r') as f:
        lstm_metrics = json.load(f)
    #return render(request, 'firstapp/model_graphs.html')
    return render(request, "firstapp/model_graphs.html", {"metrics": lstm_metrics})


def Back_Testing(request):
    if request.method == 'POST':
        days_forecast = int(request.POST.get('num_days', 1))
        stock_name = "C:XAUUSD"  
        model_day = load_model(r'firstapp\Models\gold\gold_lstm_model_day.h5', custom_objects={'mse': losses.MeanSquaredError()})

        with open(r'firstapp\Models\gold\gold_scaler.pkl', 'rb') as f:
            scaler_day = pickle.load(f)

        with open(r'firstapp\Models\gold\gold_history_day.pkl', 'rb') as f:
            history_day = pickle.load(f)

        with open(r'firstapp\Models\gold\gold_arima_day.pkl', 'rb') as f:
            arima_fit_day = pickle.load(f)

        
        # Get today's date
        today = datetime.datetime.now()

        # Convert to Eastern Standard Time (EST)
        est_timezone = pytz.timezone('Asia/Kolkata')
        est_dt = today.astimezone(est_timezone)
        minutes_value = est_dt.minute
        flor_val = minutes_value//15
        new_dt = est_dt.replace(minute=0)
        new_dt = new_dt + datetime.timedelta(minutes=15*flor_val)
        today = new_dt.replace(second=0, microsecond=0)
        new_dt = today.replace(hour=0, minute=0, second=0, microsecond=0)

        #Day wise Prediction

        data = get_candle_data(stock_name=stock_name, multiplier_val=1, timespan_val="day",
                                from_val=new_dt - timedelta(days=200),
                                to_val=today)
        data = pd.DataFrame(data)
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms').dt.date

        data = data.set_index('date')
        df_day = data.tail(60+days_forecast).copy()
        df_day_old = df_day.head(60).copy()
        df_day_future = df_day.tail(days_forecast).copy()
        close_prices = df_day_old['close'].values.reshape(-1, 1)
        scaled_close = scaler_day.fit_transform(close_prices)
        seq_length = 60
    
        # For prediction, we use the last 60 days as input sequence of shape (1, 60, 1)
        X_input = create_xsequences(scaled_close, seq_length)[-1:]

        predictions_scaled = []

        current_input = X_input  # shape (1, 60, 1)

        forecast_next_days = days_forecast

        for _ in range(forecast_next_days):
            pred_scaled = model_day.predict(current_input)[0, 0]
            predictions_scaled.append(pred_scaled)
            # Update the input by adding prediction and removing oldest
            current_input = np.append(current_input[:,1:,:], [[[pred_scaled]]], axis=1)

        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        predictions_day = scaler_day.inverse_transform(predictions_scaled).flatten()

        print("Next {} days predicted close prices:".format(forecast_next_days))

        close_prices_flat = close_prices.flatten()
        combined_day = np.concatenate([close_prices_flat, predictions_day])
        arima_pred = arima_fit_day.predict(start=arima_fit_day.loglikelihood_burn, end=len(combined_day)-1)
        day_values_list = (arima_pred[-forecast_next_days:] + predictions_day).values

        # Specify start date and frequency
        start_date = str(df_day.tail(1).index[0])
        #start_date = '2025-09-12'  # for example, tomorrow's date
        day_dates_list = pd.date_range(end=start_date, periods=len(day_values_list), freq='D')  # 'D' for daily
        print("dates",day_dates_list)
        # Convert DatetimeIndex to list of ISO formatted strings
        day_dates_list = day_dates_list.strftime('%Y-%m-%d').tolist()

        real_close_values = list(df_day_future['close'].values)[:days_forecast]

        day_dates_lists = df_day_future.index.tolist()[:days_forecast]
        print(day_dates_lists)
        # Create DataFrame
        new_df = pd.DataFrame({'day_dates_list': day_dates_lists, 'pred_close_values': day_values_list, 'real_close_values': real_close_values})
        # Create `signal` column
        new_df['signal'] = new_df['pred_close_values'].shift(1) > new_df['pred_close_values']
        new_df['signal'] = new_df['signal'].map({True: 'sell', False: 'buy'}).shift(-1)
        #new_df.loc[0, 'signal'] = 'buy'
        cash = 0
        holding = 0
        profits = []
        last_buy_price = None  # track last buy price

        for i, row in new_df.iterrows():
            price = row["real_close_values"]
            signal = row["signal"]

            # Rule 1: Buy only if not already holding
            if signal == "buy" and holding == 0:
                cash -= price
                holding += 1
                last_buy_price = price

            # Rule 2: Sell only if profitable
            elif signal == "sell" and holding > 0:
                if price > last_buy_price:   # only sell if profitable
                    cash += price
                    holding -= 1
                    last_buy_price = None

            # Rule 3: On last day, sell everything
            if i == len(new_df) - 1 and holding > 0:
                cash += holding * price
                holding = 0

            # Net profit at day end
            net_profit = cash + holding * price
            profits.append(net_profit)

        new_df["net_profit"] = profits

        #print(df_day)
        day_dates_list = new_df['day_dates_list'].tolist()
        pred_close_values = new_df['pred_close_values'].tolist()
        real_close_values = new_df['real_close_values'].tolist()

        # If you also want the signal or net_profit columns:
        signals = new_df['signal'].tolist()
        net_profits = new_df['net_profit'].tolist()

        string_dates = [d.strftime('%Y-%m-%d') for d in day_dates_list]
        print(string_dates)
        context = {
                'day_dates_list': json.dumps(string_dates),          # e.g. ['2025-09-14', '2025-09-15', ...]
                'pred_close_values': json.dumps(pred_close_values),  # e.g. [3745.16, 3750.51, ...]
                'real_close_values': json.dumps(real_close_values),       # e.g. [3637.99, 3676.73, ...]
                'net_profit': json.dumps(net_profits),         # e.g. [0, 38.74, 56.04, ...]
                'signal' : json.dumps(signals)
                }
        return render(request, "firstapp/back_testing.html",context)

        
    return render(request, "firstapp/back_testing.html")

def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = SignUpForm()
    return render(request, 'firstapp/signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')  # Change "home" to your home page's URL name
    else:
        form = AuthenticationForm()
    return render(request, 'firstapp/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')
