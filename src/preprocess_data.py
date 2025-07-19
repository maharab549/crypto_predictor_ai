import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler

def preprocess_and_feature_engineer(df_crypto, df_sentiment):
    """
    Combines crypto price data with sentiment data, engineers features,
    and prepares data for model training.
    :param df_crypto: DataFrame with crypto price data (from get_crypto_data.py)
    :param df_sentiment: DataFrame with daily sentiment data (from get_news_sentiment.py)
    :return: X_train_scaled, X_test_scaled, y_train, y_test, X_train_columns
    """
    print("Starting data preprocessing and feature engineering...")

    # Ensure crypto data index is datetime and mergeable
    df_crypto.index = pd.to_datetime(df_crypto.index)
    df_crypto.index = df_crypto.index.normalize() # Remove time component

    # Ensure sentiment data index is datetime and mergeable
    df_sentiment.index = pd.to_datetime(df_sentiment.index)
    df_sentiment.index = df_sentiment.index.normalize() # Remove time component

    # Merge crypto price data with daily sentiment data
    # Use left merge to keep all crypto data and add sentiment where available
    df_combined = df_crypto.merge(df_sentiment[["avg_sentiment"]], left_index=True, right_index=True, how=\"left\")

    # Handle missing sentiment values (e.g., days with no news). Fill with 0 (neutral).
    df_combined["avg_sentiment"].fillna(0, inplace=True)
    print("Merged crypto data with sentiment data.")

    # Ensure 'price' column is numeric
    df_combined["price"] = pd.to_numeric(df_combined["price"])

    # --- Feature Engineering: Technical Indicators ---
    # Simple Moving Average (SMA)
    df_combined["SMA_7"] = ta.trend.sma_indicator(df_combined["price"], window=7)
    df_combined["SMA_30"] = ta.trend.sma_indicator(df_combined["price"], window=30)

    # Exponential Moving Average (EMA)
    df_combined["EMA_7"] = ta.trend.ema_indicator(df_combined["price"], window=7)
    df_combined["EMA_30"] = ta.trend.ema_indicator(df_combined["price"], window=30)

    # Relative Strength Index (RSI)
    df_combined["RSI"] = ta.momentum.rsi(df_combined["price"], window=14)
    print("Calculated technical indicators.")

    # --- Feature Engineering: Lagged Features ---
    # Use past values of price, volume, and sentiment as features
    for col in ["price", "total_volume", "avg_sentiment"]:
        for i in range(1, 4): # Lag by 1, 2, and 3 days
            df_combined[f"{col}_lag_{i}"] = df_combined[col].shift(i)
    print("Created lagged features.")

    # --- Define Target Variable ---
    # Predict the next day's closing price
    df_combined["target_price"] = df_combined["price"].shift(-1)
    print("Defined target variable (next day's price).")

    # Drop any rows with NaN values that resulted from indicator/lag calculations or target shift
    df_combined.dropna(inplace=True)
    print(f"Dropped rows with NaN values. Remaining data shape: {df_combined.shape}")

    # Define features (X) and target (y)
    X = df_combined.drop("target_price", axis=1)
    y = df_combined["target_price"]

    # --- Splitting Data into Training and Testing Sets (Chronological) ---
    # Determine the split point (e.g., 80% for training, 20% for testing)
    split_index = int(len(df_combined) * 0.8)

    # Split the data chronologically
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")

    # --- Scaling Features ---
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform both training and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame with original column names for clarity
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    print("Features scaled using MinMaxScaler.")

    print("Data preprocessing and feature engineering complete.")
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns.tolist()

if __name__ == "__main__":
    # This part is for testing the script independently
    print("Running preprocess_data.py independently for testing...")
    try:
        df_crypto_test = pd.read_csv("../data/bitcoin_usd_daily_data.csv", index_col=\"timestamp\", parse_dates=True)
        df_sentiment_test = pd.read_csv("../data/daily_sentiment.csv", index_col=\"date\", parse_dates=True)

        X_train_s, X_test_s, y_train_s, y_test_s, feature_cols = preprocess_and_feature_engineer(df_crypto_test, df_sentiment_test)

        print("\nPreprocessed Data Shapes:")
        print(f"X_train_scaled: {X_train_s.shape}")
        print(f"X_test_scaled: {X_test_s.shape}")
        print(f"y_train: {y_train_s.shape}")
        print(f"y_test: {y_test_s.shape}")
        print("\nFirst 5 rows of X_train_scaled:")
        print(X_train_s.head())

    except FileNotFoundError:
        print("Error: Required data files (bitcoin_usd_daily_data.csv or daily_sentiment.csv) not found.")
        print("Please run get_crypto_data.py and get_news_sentiment.py first to generate these files.")
    except Exception as e:
        print(f"An error occurred during independent run: {e}")


