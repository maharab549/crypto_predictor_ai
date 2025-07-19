import requests
import pandas as pd
import time
import datetime

def get_coingecko_data(coin_id, vs_currency, days):
    """
    Fetches historical market data from CoinGecko API.
    :param coin_id: The ID of the coin (e.g., 'bitcoin', 'ethereum').
    :param vs_currency: The target currency of market data (e.g., 'usd', 'eur').
    :param days: Data up to number of days ago (e.g., '365' for one year, 'max' for all available data).
    :return: A pandas DataFrame containing historical market data.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': vs_currency,
        'days': days,
        'interval': 'daily' # For daily data
    }
    
    print(f"Fetching data for {coin_id} in {vs_currency} for {days} days...")
    response = requests.get(url, params=params)
    response.raise_for_status() # Raise an exception for HTTP errors
    data = response.json()
    
    # Extract prices, market caps, and total volumes
    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
    total_volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'total_volume'])
    
    # Convert timestamps to datetime objects
    prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
    market_caps['timestamp'] = pd.to_datetime(market_caps['timestamp'], unit='ms')
    total_volumes['timestamp'] = pd.to_datetime(total_volumes['timestamp'], unit='ms')
    
    # Set timestamp as index
    prices.set_index('timestamp', inplace=True)
    market_caps.set_index('timestamp', inplace=True)
    total_volumes.set_index('timestamp', inplace=True)
    
    # Merge dataframes
    df = prices.merge(market_caps, on='timestamp').merge(total_volumes, on='timestamp')
    
    print("Data fetched successfully.")
    return df

if __name__ == "__main__":
    coin = 'bitcoin'
    currency = 'usd'
    # Fetch data for the last 365 days (1 year)
    df_btc = get_coingecko_data(coin, currency, '365')
    
    # Save to CSV
    output_filename = f'../data/{coin}_{currency}_daily_data.csv'
    df_btc.to_csv(output_filename)
    print(f"Data saved to {output_filename}")

    # Display the first few rows of the dataframe
    print("\nFirst 5 rows of the data:")
    print(df_btc.head())

    # Display basic info
    print("\nDataframe Info:")
    df_btc.info()

    # Example of fetching data for a different coin (Ethereum for 90 days)
    # time.sleep(60) # Be mindful of API rate limits, wait before next request
    # df_eth = get_coingecko_data('ethereum', 'usd', '90')
    # df_eth.to_csv('../data/ethereum_usd_daily_data.csv')
    # print(f"Data saved to ../data/ethereum_usd_daily_data.csv")


