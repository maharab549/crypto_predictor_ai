import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib # For saving models

# Function to train and evaluate Linear Regression model
def train_evaluate_linear_regression(X_train, y_train, X_test, y_test):
    print("\n--- Training Linear Regression Model ---")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Linear Regression MSE: {mse:.4f}")
    print(f"Linear Regression RMSE: {rmse:.4f}")
    print(f"Linear Regression MAE: {mae:.4f}")
    
    # Save the model
    joblib.dump(linear_model, "../models/linear_regression_model.pkl")
    print("Linear Regression model saved to ../models/linear_regression_model.pkl")

    return y_pred, mse, rmse, mae

# Function to train and evaluate LSTM model
def train_evaluate_lstm(X_train, y_train, X_test, y_test, features_count):
    print("\n--- Training LSTM Model ---")

    # Reshape data for LSTM: [samples, timesteps, features]
    # For our current setup, timesteps will be 1 as we are using single-day features
    X_train_lstm = X_train.values.reshape(X_train.shape[0], 1, features_count)
    X_test_lstm = X_test.values.reshape(X_test.shape[0], 1, features_count)

    lstm_model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(1, features_count)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1) # Output layer for predicting a single price
    ])

    lstm_model.compile(optimizer=\'adam\', loss=\'mean_squared_error\')

    early_stopping = EarlyStopping(monitor=\'val_loss\', patience=10, restore_best_weights=True)

    history = lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, 
                             validation_split=0.2, callbacks=[early_stopping], verbose=1)
    
    # Save the model
    lstm_model.save("../models/lstm_model.h5")
    print("LSTM model saved to ../models/lstm_model.h5")

    y_pred = lstm_model.predict(X_test_lstm)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"LSTM MSE: {mse:.4f}")
    print(f"LSTM RMSE: {rmse:.4f}")
    print(f"LSTM MAE: {mae:.4f}")

    return y_pred, mse, rmse, mae, history

# Function to plot results
def plot_predictions(y_test, y_pred, title, filename):
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test, label=\'Actual Prices\', color=\'blue\')
    plt.plot(y_test.index, y_pred, label=\'Predicted Prices\', color=\'red\', linestyle=\'--\')
    plt.title(title)
    plt.xlabel(\'Date\')
    plt.ylabel(\'Price (USD)\')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../data/{filename}.png")
    plt.close()
    print(f"Plot saved to ../data/{filename}.png")

def plot_loss_history(history, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history[\\'loss\\'], label=\\'Train Loss\\')
    plt.plot(history.history[\\'val_loss\\'], label=\\'Validation Loss\\')
    plt.title(title)
    plt.xlabel(\'Epoch\')
    plt.ylabel(\'Loss\')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../data/{filename}.png")
    plt.close()
    print(f"Plot saved to ../data/{filename}.png")

if __name__ == "__main__":
    # This part is for testing the script independently
    print("Running train_models.py independently for testing...")
    try:
        # Load preprocessed data (assuming preprocess_data.py has been run)
        X_train_scaled = pd.read_csv("../data/X_train_scaled.csv", index_col=0, parse_dates=True)
        X_test_scaled = pd.read_csv("../data/X_test_scaled.csv", index_col=0, parse_dates=True)
        y_train = pd.read_csv("../data/y_train.csv", index_col=0, parse_dates=True).squeeze()
        y_test = pd.read_csv("../data/y_test.csv", index_col=0, parse_dates=True).squeeze()

        features_count = X_train_scaled.shape[1]

        # Train and evaluate Linear Regression
        y_pred_lr, mse_lr, rmse_lr, mae_lr = train_evaluate_linear_regression(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        plot_predictions(y_test, y_pred_lr, 
                         "Bitcoin Price Prediction - Linear Regression", 
                         "linear_regression_predictions")

        # Train and evaluate LSTM
        y_pred_lstm, mse_lstm, rmse_lstm, mae_lstm, history_lstm = train_evaluate_lstm(
            X_train_scaled, y_train, X_test_scaled, y_test, features_count
        )
        plot_predictions(y_test, y_pred_lstm, 
                         "Bitcoin Price Prediction - LSTM", 
                         "lstm_predictions")
        plot_loss_history(history_lstm, 
                          "LSTM Model Loss History", 
                          "lstm_loss_history")

    except FileNotFoundError:
        print("Error: Required preprocessed data files (X_train_scaled.csv, etc.) not found.")
        print("Please run preprocess_data.py first to generate these files.")
    except Exception as e:
        print(f"An error occurred during independent run: {e}")



