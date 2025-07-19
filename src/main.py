#!/usr/bin/env python3
"""
Crypto Predictor AI - Main Pipeline
This script orchestrates the entire crypto prediction pipeline:
1. Data collection (crypto prices and news sentiment)
2. Data preprocessing and feature engineering
3. Model training and evaluation
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Import our custom modules
from get_crypto_data import get_coingecko_data
from get_news_sentiment import create_dummy_news_data, aggregate_daily_sentiment
from preprocess_data import preprocess_and_feature_engineer
from train_models import train_evaluate_linear_regression, train_evaluate_lstm, plot_predictions, plot_loss_history

def main():
    print("=" * 60)
    print("CRYPTO PREDICTOR AI - FULL PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # Create necessary directories
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../models", exist_ok=True)
    
    # Step 1: Collect Crypto Price Data
    print("\n" + "=" * 40)
    print("STEP 1: COLLECTING CRYPTO PRICE DATA")
    print("=" * 40)
    
    try:
        df_crypto = get_coingecko_data('bitcoin', 'usd', '365')
        df_crypto.to_csv('../data/bitcoin_usd_daily_data.csv')
        print("✓ Crypto price data collected and saved")
    except Exception as e:
        print(f"✗ Error collecting crypto data: {e}")
        return
    
    # Step 2: Collect News Sentiment Data
    print("\n" + "=" * 40)
    print("STEP 2: COLLECTING NEWS SENTIMENT DATA")
    print("=" * 40)
    
    try:
        # Create dummy news data (in real implementation, this would use news APIs)
        df_news = create_dummy_news_data(365)
        daily_sentiment = aggregate_daily_sentiment(df_news)
        
        df_news.to_csv('../data/news_articles.csv', index=False)
        daily_sentiment.to_csv('../data/daily_sentiment.csv')
        print("✓ News sentiment data collected and saved")
    except Exception as e:
        print(f"✗ Error collecting news sentiment data: {e}")
        return
    
    # Step 3: Preprocess Data and Engineer Features
    print("\n" + "=" * 40)
    print("STEP 3: PREPROCESSING DATA & FEATURE ENGINEERING")
    print("=" * 40)
    
    try:
        X_train_scaled, X_test_scaled, y_train, y_test, feature_columns = preprocess_and_feature_engineer(
            df_crypto, daily_sentiment
        )
        
        # Save preprocessed data for later use
        X_train_scaled.to_csv('../data/X_train_scaled.csv')
        X_test_scaled.to_csv('../data/X_test_scaled.csv')
        y_train.to_csv('../data/y_train.csv')
        y_test.to_csv('../data/y_test.csv')
        
        print("✓ Data preprocessing and feature engineering completed")
        print(f"  - Training samples: {len(X_train_scaled)}")
        print(f"  - Testing samples: {len(X_test_scaled)}")
        print(f"  - Features: {len(feature_columns)}")
    except Exception as e:
        print(f"✗ Error in data preprocessing: {e}")
        return
    
    # Step 4: Train and Evaluate Models
    print("\n" + "=" * 40)
    print("STEP 4: TRAINING AND EVALUATING MODELS")
    print("=" * 40)
    
    try:
        # Train Linear Regression
        y_pred_lr, mse_lr, rmse_lr, mae_lr = train_evaluate_linear_regression(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Train LSTM
        y_pred_lstm, mse_lstm, rmse_lstm, mae_lstm, history_lstm = train_evaluate_lstm(
            X_train_scaled, y_train, X_test_scaled, y_test, len(feature_columns)
        )
        
        print("✓ Model training completed")
        
        # Generate plots
        plot_predictions(y_test, y_pred_lr, 
                        "Bitcoin Price Prediction - Linear Regression", 
                        "linear_regression_predictions")
        
        plot_predictions(y_test, y_pred_lstm, 
                        "Bitcoin Price Prediction - LSTM", 
                        "lstm_predictions")
        
        plot_loss_history(history_lstm, 
                         "LSTM Model Loss History", 
                         "lstm_loss_history")
        
        print("✓ Prediction plots generated")
        
    except Exception as e:
        print(f"✗ Error in model training: {e}")
        return
    
    # Step 5: Summary Report
    print("\n" + "=" * 40)
    print("STEP 5: FINAL RESULTS SUMMARY")
    print("=" * 40)
    
    print(f"Linear Regression Performance:")
    print(f"  - RMSE: ${rmse_lr:,.2f}")
    print(f"  - MAE:  ${mae_lr:,.2f}")
    
    print(f"\nLSTM Performance:")
    print(f"  - RMSE: ${rmse_lstm:,.2f}")
    print(f"  - MAE:  ${mae_lstm:,.2f}")
    
    # Determine better model
    better_model = "Linear Regression" if rmse_lr < rmse_lstm else "LSTM"
    print(f"\nBetter performing model: {better_model}")
    
    print(f"\nFiles generated:")
    print(f"  - Data: ../data/")
    print(f"  - Models: ../models/")
    print(f"  - Plots: ../data/*.png")
    
    print(f"\nPipeline completed successfully at: {datetime.now()}")
    print("=" * 60)

if __name__ == "__main__":
    main()

