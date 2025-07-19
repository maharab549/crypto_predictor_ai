# Crypto Predictor AI 🚀

A comprehensive machine learning project that predicts cryptocurrency prices using historical market data and news sentiment analysis.

## 🌟 Features

- **Real-time Data Collection**: Fetches historical cryptocurrency data from CoinGecko API
- **News Sentiment Analysis**: Analyzes news sentiment using state-of-the-art NLP models
- **Technical Indicators**: Implements popular technical analysis indicators (SMA, EMA, RSI)
- **Multiple ML Models**: Includes both Linear Regression and LSTM neural networks
- **Feature Engineering**: Creates lagged features and combines multiple data sources
- **Visualization**: Generates prediction plots and model performance charts
- **Modular Design**: Clean, well-documented code structure

## 📁 Project Structure

```
crypto_predictor_ai/
├── src/
│   ├── main.py                 # Main pipeline orchestrator
│   ├── get_crypto_data.py      # Cryptocurrency data collection
│   ├── get_news_sentiment.py   # News sentiment analysis
│   ├── preprocess_data.py      # Data preprocessing & feature engineering
│   └── train_models.py         # Model training & evaluation
├── data/                       # Generated data files
├── models/                     # Saved trained models
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 32GB RAM (recommended for LSTM training)
- GPU support (optional, for faster LSTM training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/crypto_predictor_ai.git
   cd crypto_predictor_ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv crypto_env
   
   # On Windows:
   crypto_env\Scripts\activate
   
   # On macOS/Linux:
   source crypto_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

#### Option 1: Run the Complete Pipeline

```bash
cd src
python main.py
```

This will execute the entire pipeline:
1. Collect Bitcoin price data (last 365 days)
2. Generate news sentiment data
3. Preprocess and engineer features
4. Train both Linear Regression and LSTM models
5. Generate prediction plots and performance metrics

#### Option 2: Run Individual Components

**Collect crypto data:**
```bash
cd src
python get_crypto_data.py
```

**Generate news sentiment data:**
```bash
cd src
python get_news_sentiment.py
```

**Preprocess data:**
```bash
cd src
python preprocess_data.py
```

**Train models:**
```bash
cd src
python train_models.py
```

## 📊 Features Included

### Technical Indicators
- **Simple Moving Average (SMA)**: 7-day and 30-day periods
- **Exponential Moving Average (EMA)**: 7-day and 30-day periods
- **Relative Strength Index (RSI)**: 14-day period

### Sentiment Features
- **Daily Average Sentiment**: Aggregated from news articles
- **News Volume**: Number of articles per day
- **Sentiment Volatility**: Standard deviation of daily sentiment

### Lagged Features
- Price, volume, and sentiment values from 1-3 days ago

## 🤖 Models

### 1. Linear Regression
- **Purpose**: Baseline model for comparison
- **Features**: All engineered features
- **Output**: Next day's closing price

### 2. LSTM Neural Network
- **Architecture**: 2 LSTM layers with dropout
- **Purpose**: Capture temporal dependencies
- **Features**: Same as Linear Regression
- **Output**: Next day's closing price

## 📈 Performance Metrics

The models are evaluated using:
- **RMSE (Root Mean Square Error)**: Primary metric for model comparison
- **MAE (Mean Absolute Error)**: Average prediction error in dollars
- **MSE (Mean Square Error)**: Squared error metric

## 🔧 Configuration

### News API Integration (Optional)

To use real news data instead of dummy data:

1. Get a free API key from [NewsAPI.org](https://newsapi.org/)
2. Set the environment variable:
   ```bash
   # Windows
   set NEWS_API_KEY=your_api_key_here
   
   # macOS/Linux
   export NEWS_API_KEY=your_api_key_here
   ```

### GPU Support for LSTM

For AMD GPUs (like RX 7600S):
- Install ROCm platform for TensorFlow GPU support
- The model will automatically fall back to CPU if GPU is not available

## 📋 Output Files

After running the pipeline, you'll find:

### Data Files (`data/` directory)
- `bitcoin_usd_daily_data.csv`: Raw cryptocurrency data
- `news_articles.csv`: News articles with sentiment scores
- `daily_sentiment.csv`: Aggregated daily sentiment
- `X_train_scaled.csv`, `X_test_scaled.csv`: Preprocessed features
- `y_train.csv`, `y_test.csv`: Target variables

### Model Files (`models/` directory)
- `linear_regression_model.pkl`: Trained Linear Regression model
- `lstm_model.h5`: Trained LSTM model

### Visualization Files (`data/` directory)
- `linear_regression_predictions.png`: Linear Regression predictions plot
- `lstm_predictions.png`: LSTM predictions plot
- `lstm_loss_history.png`: LSTM training loss history

## 🛠️ Customization

### Adding New Cryptocurrencies

Modify `get_crypto_data.py`:
```python
# Change the coin parameter
coin = 'ethereum'  # or 'cardano', 'solana', etc.
```

### Adding More Technical Indicators

In `preprocess_data.py`, add new indicators:
```python
# Example: Bollinger Bands
df_combined["BB_upper"] = ta.volatility.bollinger_hband(df_combined["price"])
df_combined["BB_lower"] = ta.volatility.bollinger_lband(df_combined["price"])
```

### Tuning Model Parameters

In `train_models.py`, modify LSTM architecture:
```python
# Example: Add more layers or change units
LSTM(units=100, return_sequences=True),  # Increase units
LSTM(units=100, return_sequences=True),  # Add another layer
```

## ⚠️ Important Notes

### Limitations
- **Market Volatility**: Cryptocurrency markets are highly volatile and unpredictable
- **No Financial Advice**: This is for educational purposes only, not financial advice
- **Data Quality**: Predictions depend heavily on data quality and market conditions
- **Overfitting Risk**: Always validate models on out-of-sample data

### Best Practices
- **Paper Trading**: Test strategies with paper trading before real money
- **Risk Management**: Never invest more than you can afford to lose
- **Continuous Learning**: Markets evolve; models need regular updates
- **Diversification**: Don't rely on a single prediction model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CoinGecko](https://www.coingecko.com/) for cryptocurrency data API
- [NewsAPI](https://newsapi.org/) for news data
- [Hugging Face Transformers](https://huggingface.co/transformers/) for sentiment analysis
- [TA-Lib](https://github.com/mrjbq7/ta-lib) for technical analysis indicators

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/crypto_predictor_ai/issues) page
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Disclaimer**: This project is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

