import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os

# For sentiment analysis
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers torch")

def get_news_api_data(api_key, query, from_date, to_date, language='en', sort_by='publishedAt'):
    """
    Fetches news data from NewsAPI.org
    :param api_key: Your NewsAPI.org API key
    :param query: Search query (e.g., 'bitcoin OR cryptocurrency')
    :param from_date: Start date (YYYY-MM-DD format)
    :param to_date: End date (YYYY-MM-DD format)
    :param language: Language code (default: 'en')
    :param sort_by: Sort by (publishedAt, relevancy, popularity)
    :return: List of articles
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'language': language,
        'sortBy': sort_by,
        'apiKey': api_key,
        'pageSize': 100  # Maximum articles per request
    }
    
    print(f"Fetching news for query: '{query}' from {from_date} to {to_date}")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        articles = data.get('articles', [])
        print(f"Fetched {len(articles)} articles")
        return articles
    else:
        print(f"Error fetching news: {response.status_code} - {response.text}")
        return []

def analyze_sentiment(text):
    """
    Analyzes sentiment of given text using transformers pipeline
    :param text: Text to analyze
    :return: Sentiment score (-1 to 1, where -1 is negative, 1 is positive)
    """
    if not TRANSFORMERS_AVAILABLE:
        # Return random sentiment if transformers not available (for demo purposes)
        return np.random.uniform(-0.5, 0.5)
    
    try:
        # Initialize sentiment analyzer (cached after first use)
        if not hasattr(analyze_sentiment, 'analyzer'):
            analyze_sentiment.analyzer = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
        
        result = analyze_sentiment.analyzer(text[:512])  # Limit text length
        
        # Extract scores for positive and negative
        scores = {item['label']: item['score'] for item in result[0]}
        
        # Convert to -1 to 1 scale
        if 'POSITIVE' in scores and 'NEGATIVE' in scores:
            return scores['POSITIVE'] - scores['NEGATIVE']
        else:
            return 0.0
            
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return 0.0

def create_dummy_news_data(days=365):
    """
    Creates dummy news data for demonstration purposes
    :param days: Number of days of dummy data to create
    :return: DataFrame with dummy news data
    """
    print(f"Creating dummy news data for {days} days...")
    
    # Sample crypto-related headlines (mix of positive, negative, neutral)
    sample_headlines = [
        "Bitcoin reaches new all-time high as institutional adoption grows",
        "Major cryptocurrency exchange reports security breach",
        "Ethereum upgrade promises faster transactions and lower fees",
        "Regulatory uncertainty clouds cryptocurrency market outlook",
        "Tesla announces acceptance of Bitcoin payments",
        "Central bank warns against cryptocurrency investments",
        "Blockchain technology revolutionizes supply chain management",
        "Cryptocurrency market experiences significant volatility",
        "Major bank launches cryptocurrency trading services",
        "Environmental concerns raised over Bitcoin mining",
        "DeFi protocols gain mainstream attention",
        "Government considers cryptocurrency regulation framework",
        "NFT market shows explosive growth",
        "Cryptocurrency adoption increases in developing countries",
        "Market analysts predict continued crypto growth"
    ]
    
    # Generate dummy data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    news_data = []
    current_date = start_date
    
    while current_date <= end_date:
        # Generate 1-5 articles per day
        num_articles = np.random.randint(1, 6)
        
        for _ in range(num_articles):
            headline = np.random.choice(sample_headlines)
            # Add some variation to headlines
            if np.random.random() > 0.7:
                headline += f" - Market Update {current_date.strftime('%Y-%m-%d')}"
            
            sentiment_score = analyze_sentiment(headline)
            
            news_data.append({
                'date': current_date.date(),
                'headline': headline,
                'sentiment_score': sentiment_score,
                'source': 'DummyNews'
            })
        
        current_date += timedelta(days=1)
    
    df = pd.DataFrame(news_data)
    print(f"Created {len(df)} dummy news articles")
    return df

def process_news_articles(articles):
    """
    Processes news articles and performs sentiment analysis
    :param articles: List of articles from NewsAPI
    :return: DataFrame with processed articles
    """
    processed_data = []
    
    for article in articles:
        # Extract relevant information
        title = article.get('title', '')
        description = article.get('description', '')
        published_at = article.get('publishedAt', '')
        source = article.get('source', {}).get('name', 'Unknown')
        
        # Combine title and description for sentiment analysis
        text_for_analysis = f"{title}. {description}" if description else title
        
        # Perform sentiment analysis
        sentiment_score = analyze_sentiment(text_for_analysis)
        
        # Parse date
        try:
            date = datetime.fromisoformat(published_at.replace('Z', '+00:00')).date()
        except:
            date = datetime.now().date()
        
        processed_data.append({
            'date': date,
            'headline': title,
            'description': description,
            'sentiment_score': sentiment_score,
            'source': source
        })
    
    return pd.DataFrame(processed_data)

def aggregate_daily_sentiment(df_news):
    """
    Aggregates news sentiment by day
    :param df_news: DataFrame with news articles and sentiment scores
    :return: DataFrame with daily aggregated sentiment
    """
    daily_sentiment = df_news.groupby('date').agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    daily_sentiment.columns = ['date', 'avg_sentiment', 'sentiment_std', 'news_count']
    
    # Fill NaN standard deviations with 0
    daily_sentiment['sentiment_std'].fillna(0, inplace=True)
    
    # Convert date to datetime and set as index
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    daily_sentiment.set_index('date', inplace=True)
    
    return daily_sentiment

if __name__ == "__main__":
    # Check if NewsAPI key is available
    news_api_key = os.getenv('NEWS_API_KEY')
    
    if news_api_key:
        print("NewsAPI key found. Fetching real news data...")
        
        # Calculate date range (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Fetch news articles
        articles = get_news_api_data(
            api_key=news_api_key,
            query='bitcoin OR cryptocurrency OR blockchain',
            from_date=start_date.strftime('%Y-%m-%d'),
            to_date=end_date.strftime('%Y-%m-%d')
        )
        
        if articles:
            # Process articles
            df_news = process_news_articles(articles)
        else:
            print("No articles fetched. Creating dummy data...")
            df_news = create_dummy_news_data(30)
    else:
        print("No NewsAPI key found. Creating dummy news data...")
        print("To use real news data, set NEWS_API_KEY environment variable")
        print("Get your free API key from: https://newsapi.org/")
        df_news = create_dummy_news_data()
    
    # Aggregate daily sentiment
    daily_sentiment = aggregate_daily_sentiment(df_news)
    
    # Save data
    df_news.to_csv('../data/news_articles.csv', index=False)
    daily_sentiment.to_csv('../data/daily_sentiment.csv')
    
    print(f"\nNews data saved to ../data/news_articles.csv")
    print(f"Daily sentiment data saved to ../data/daily_sentiment.csv")
    
    print("\nDaily sentiment summary:")
    print(daily_sentiment.head())
    print(f"\nSentiment statistics:")
    print(daily_sentiment['avg_sentiment'].describe())

