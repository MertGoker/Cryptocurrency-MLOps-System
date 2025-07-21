import numpy as np
import pandas as pd
from transformers import pipeline
import requests
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment analyzer for cryptocurrency news and social media
    """
    
    def __init__(self):
        self.sentiment_pipeline = None
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.initialize_sentiment_model()
    
    def initialize_sentiment_model(self):
        """
        Initialize the sentiment analysis model
        """
        try:
            # Use a pre-trained sentiment analysis model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            logger.info("Sentiment analysis model initialized successfully")
        except Exception as e:
            logger.warning(f"Could not load sentiment model: {str(e)}. Using mock analysis.")
            self.sentiment_pipeline = None
    
    def get_news_data(self, symbol, source="all", limit=100):
        """
        Fetch news data for cryptocurrency
        """
        try:
            # Mock news data (in real implementation, this would fetch from news APIs)
            crypto_name = symbol.split('-')[0] if '-' in symbol else symbol
            
            mock_news = [
                {
                    "title": f"{crypto_name} adoption increases in major markets",
                    "content": f"Recent developments show growing adoption of {crypto_name} in institutional markets.",
                    "source": "CryptoNews",
                    "published_at": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "url": f"https://example.com/news/{crypto_name.lower()}-adoption"
                },
                {
                    "title": f"Regulatory concerns about {crypto_name}",
                    "content": f"Regulators are discussing new policies for {crypto_name} trading.",
                    "source": "FinanceDaily",
                    "published_at": (datetime.now() - timedelta(hours=4)).isoformat(),
                    "url": f"https://example.com/news/{crypto_name.lower()}-regulation"
                },
                {
                    "title": f"{crypto_name} technical analysis shows bullish pattern",
                    "content": f"Technical indicators suggest a bullish trend for {crypto_name}.",
                    "source": "TradingView",
                    "published_at": (datetime.now() - timedelta(hours=6)).isoformat(),
                    "url": f"https://example.com/analysis/{crypto_name.lower()}-technical"
                },
                {
                    "title": f"Market volatility affects {crypto_name} trading",
                    "content": f"Recent market volatility has impacted {crypto_name} trading volumes.",
                    "source": "MarketWatch",
                    "published_at": (datetime.now() - timedelta(hours=8)).isoformat(),
                    "url": f"https://example.com/markets/{crypto_name.lower()}-volatility"
                },
                {
                    "title": f"{crypto_name} partnership with major tech company",
                    "content": f"A new partnership could boost {crypto_name} adoption significantly.",
                    "source": "TechCrunch",
                    "published_at": (datetime.now() - timedelta(hours=10)).isoformat(),
                    "url": f"https://example.com/tech/{crypto_name.lower()}-partnership"
                }
            ]
            
            return mock_news[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching news data: {str(e)}")
            return []
    
    def analyze_text_sentiment(self, text):
        """
        Analyze sentiment of a single text
        """
        try:
            if self.sentiment_pipeline:
                # Use the actual sentiment model
                result = self.sentiment_pipeline(text[:512])  # Limit text length
                scores = result[0]
                
                # Map scores to sentiment
                sentiment_scores = {score['label']: score['score'] for score in scores}
                
                # Determine overall sentiment
                if sentiment_scores.get('positive', 0) > sentiment_scores.get('negative', 0):
                    sentiment = 'positive'
                    score = sentiment_scores.get('positive', 0)
                elif sentiment_scores.get('negative', 0) > sentiment_scores.get('positive', 0):
                    sentiment = 'negative'
                    score = sentiment_scores.get('negative', 0)
                else:
                    sentiment = 'neutral'
                    score = sentiment_scores.get('neutral', 0)
                
                return sentiment, score
            else:
                # Mock sentiment analysis
                return self.mock_sentiment_analysis(text)
                
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {str(e)}")
            return 'neutral', 0.5
    
    def mock_sentiment_analysis(self, text):
        """
        Mock sentiment analysis for demonstration
        """
        text_lower = text.lower()
        
        # Simple keyword-based sentiment analysis
        positive_words = ['bullish', 'adoption', 'partnership', 'growth', 'positive', 'increase', 'success']
        negative_words = ['bearish', 'regulation', 'concern', 'volatility', 'negative', 'decrease', 'risk']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive', min(0.9, 0.6 + positive_count * 0.1)
        elif negative_count > positive_count:
            return 'negative', min(0.9, 0.6 + negative_count * 0.1)
        else:
            return 'neutral', 0.5
    
    def analyze(self, symbol, source="all", limit=100):
        """
        Perform comprehensive sentiment analysis
        """
        try:
            logger.info(f"Analyzing sentiment for {symbol}")
            
            # Get news data
            news_data = self.get_news_data(symbol, source, limit)
            
            if not news_data:
                return self.get_default_sentiment_response(symbol)
            
            # Analyze sentiment for each news item
            sentiment_results = []
            for news in news_data:
                # Combine title and content for analysis
                text = f"{news['title']} {news['content']}"
                sentiment, score = self.analyze_text_sentiment(text)
                
                sentiment_results.append({
                    'title': news['title'],
                    'sentiment': sentiment,
                    'score': score,
                    'source': news['source'],
                    'published_at': news['published_at'],
                    'url': news['url']
                })
            
            # Calculate overall sentiment
            overall_sentiment = self.calculate_overall_sentiment(sentiment_results)
            
            # Calculate sentiment distribution
            distribution = self.calculate_sentiment_distribution(sentiment_results)
            
            # Calculate confidence
            confidence = self.calculate_confidence(sentiment_results)
            
            return {
                'overall_sentiment': overall_sentiment,
                'distribution': distribution,
                'recent_news': sentiment_results[:10],  # Return top 10 news items
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return self.get_default_sentiment_response(symbol)
    
    def calculate_overall_sentiment(self, sentiment_results):
        """
        Calculate overall sentiment score
        """
        try:
            if not sentiment_results:
                return 0.0
            
            # Weight by sentiment score and recency
            weighted_scores = []
            for result in sentiment_results:
                # Convert sentiment to numeric score
                if result['sentiment'] == 'positive':
                    sentiment_score = result['score']
                elif result['sentiment'] == 'negative':
                    sentiment_score = -result['score']
                else:
                    sentiment_score = 0
                
                # Apply time decay (more recent news has higher weight)
                published_time = datetime.fromisoformat(result['published_at'].replace('Z', '+00:00'))
                hours_ago = (datetime.now() - published_time).total_seconds() / 3600
                time_weight = max(0.1, 1 - (hours_ago / 24))  # Decay over 24 hours
                
                weighted_scores.append(sentiment_score * time_weight)
            
            # Calculate weighted average
            overall_sentiment = np.mean(weighted_scores)
            
            # Normalize to [-1, 1] range
            return max(-1.0, min(1.0, overall_sentiment))
            
        except Exception as e:
            logger.error(f"Error calculating overall sentiment: {str(e)}")
            return 0.0
    
    def calculate_sentiment_distribution(self, sentiment_results):
        """
        Calculate distribution of sentiments
        """
        try:
            distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
            
            for result in sentiment_results:
                sentiment = result['sentiment']
                if sentiment in distribution:
                    distribution[sentiment] += 1
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error calculating sentiment distribution: {str(e)}")
            return {'positive': 0, 'neutral': 0, 'negative': 0}
    
    def calculate_confidence(self, sentiment_results):
        """
        Calculate confidence in sentiment analysis
        """
        try:
            if not sentiment_results:
                return 0.5
            
            # Calculate average sentiment score
            scores = [result['score'] for result in sentiment_results]
            avg_score = np.mean(scores)
            
            # Calculate consistency (lower variance = higher confidence)
            variance = np.var(scores)
            consistency = max(0.1, 1 - variance)
            
            # Combine factors
            confidence = (avg_score + consistency) / 2
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def get_default_sentiment_response(self, symbol):
        """
        Return default sentiment response when analysis fails
        """
        return {
            'overall_sentiment': 0.0,
            'distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'recent_news': [],
            'confidence': 0.5
        }
    
    def get_sentiment_trend(self, symbol, days=7):
        """
        Get sentiment trend over time with enhanced analysis
        """
        try:
            # Generate dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
            
            # Generate realistic sentiment data with trends
            np.random.seed(hash(symbol) % 1000)  # Consistent results for same symbol
            
            # Create base sentiment with trend
            base_sentiment = np.random.uniform(-0.2, 0.3)
            trend = np.linspace(-0.1, 0.1, days)  # Gradual trend
            noise = np.random.normal(0, 0.15, days)  # Random noise
            sentiment_scores = base_sentiment + trend + noise
            
            # Ensure scores are within [-1, 1] range
            sentiment_scores = np.clip(sentiment_scores, -1, 1)
            
            # Calculate additional metrics
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_volatility = np.std(sentiment_scores)
            max_sentiment = np.max(sentiment_scores)
            min_sentiment = np.min(sentiment_scores)
            
            # Determine trend direction
            if len(sentiment_scores) > 1:
                recent_trend = sentiment_scores[-1] - sentiment_scores[0]
                if recent_trend > 0.1:
                    trend_direction = 'bullish'
                elif recent_trend < -0.1:
                    trend_direction = 'bearish'
                else:
                    trend_direction = 'sideways'
            else:
                trend_direction = 'stable'
            
            # Categorize sentiment distribution
            positive_days = np.sum(sentiment_scores > 0.3)
            negative_days = np.sum(sentiment_scores < -0.3)
            neutral_days = days - positive_days - negative_days
            
            return {
                'dates': dates,
                'sentiments': sentiment_scores.tolist(),
                'trend': trend_direction,
                'metrics': {
                    'average_sentiment': float(avg_sentiment),
                    'volatility': float(sentiment_volatility),
                    'max_sentiment': float(max_sentiment),
                    'min_sentiment': float(min_sentiment),
                    'positive_days': int(positive_days),
                    'negative_days': int(negative_days),
                    'neutral_days': int(neutral_days)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment trend: {str(e)}")
            return {
                'dates': [], 
                'sentiments': [], 
                'trend': 'unknown',
                'metrics': {}
            } 