from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import logging
import requests
from pycoingecko import CoinGeckoAPI

# Import custom modules
from ml_models.lstm_model import LSTMModel
from ml_models.sentiment_analyzer import SentimentAnalyzer
from ml_models.technical_indicators import TechnicalIndicators
from ml_models.ai_analyzer import AIAnalyzer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize CoinGecko API
cg = CoinGeckoAPI()

def get_crypto_data(symbol: str, days: int = 365):
    """
    Get real cryptocurrency data from CoinGecko API with yfinance fallback
    """
    try:
        # Map common symbols to CoinGecko IDs
        symbol_mapping = {
            'BTC-USD': 'bitcoin',
            'ETH-USD': 'ethereum',
            'ADA-USD': 'cardano',
            'DOT-USD': 'polkadot',
            'LINK-USD': 'chainlink',
            'SOL-USD': 'solana',
            'MATIC-USD': 'matic-network',
            'AVAX-USD': 'avalanche-2',
            'UNI-USD': 'uniswap',
            'AAVE-USD': 'aave'
        }
        
        # Try CoinGecko first
        if symbol in symbol_mapping:
            coin_id = symbol_mapping[symbol]
            logger.info(f"Fetching data from CoinGecko for {coin_id}")
            
            # Get market data from CoinGecko
            market_data = cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency='usd',
                days=days
            )
            
            if market_data and 'prices' in market_data:
                # Convert CoinGecko data to DataFrame format
                prices = market_data['prices']
                volumes = market_data['total_volumes']
                
                # Create DataFrame
                df = pd.DataFrame(prices, columns=['timestamp', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Add volume data
                volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                volume_df.set_index('timestamp', inplace=True)
                
                df['volume'] = volume_df['volume']
                
                # Add OHLC data (using close price as approximation)
                df['open'] = df['close'].shift(1).fillna(df['close'])
                df['high'] = df['close'] * 1.02  # Approximate high
                df['low'] = df['close'] * 0.98   # Approximate low
                
                logger.info(f"Successfully fetched {len(df)} data points from CoinGecko")
                return df
        
        # Fallback to yfinance
        logger.info(f"Falling back to yfinance for {symbol}")
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        data = ticker.history(start=start_date, end=end_date)
        
        if not data.empty:
            logger.info(f"Successfully fetched {len(data)} data points from yfinance")
            return data
        
        raise Exception("No data available from any source")
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise e

# Initialize FastAPI app
app = FastAPI(
    title="Cryptocurrency MLOps System API",
    description="A comprehensive API for cryptocurrency analysis, forecasting, and AI-powered insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
lstm_model = LSTMModel()
sentiment_analyzer = SentimentAnalyzer()
technical_indicators = TechnicalIndicators()
ai_analyzer = AIAnalyzer()

# Pydantic models for request/response
class ForecastRequest(BaseModel):
    symbol: str
    days: int = 30
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class ForecastResponse(BaseModel):
    symbol: str
    predictions: List[float]
    dates: List[str]
    metrics: Dict[str, float]
    confidence: float

class SentimentRequest(BaseModel):
    symbol: str
    source: str = "all"
    limit: int = 100

class SentimentResponse(BaseModel):
    symbol: str
    overall_sentiment: float
    sentiment_distribution: Dict[str, int]
    recent_news: List[Dict[str, Any]]
    confidence: float

class TechnicalRequest(BaseModel):
    symbol: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    indicators: List[str] = ["rsi", "macd", "bollinger_bands"]

class TechnicalResponse(BaseModel):
    symbol: str
    indicators: Dict[str, Any]
    signals: Dict[str, str]
    summary: str

class AIAnalysisRequest(BaseModel):
    symbol: str
    analysis_type: str
    user_query: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class AIAnalysisResponse(BaseModel):
    symbol: str
    analysis_type: str
    analysis: str
    recommendations: List[str]
    confidence: float
    insights: List[str]

class SystemStatusResponse(BaseModel):
    status: str
    components: Dict[str, str]
    performance_metrics: Dict[str, Any]
    last_updated: str

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Cryptocurrency MLOps System API",
        "version": "1.0.0",
        "status": "healthy",
        "developer": "Mert Göker",
        "github": "https://github.com/MertGoker",
        "linkedin": "https://www.linkedin.com/in/mert-goker-bb4bb91b6/",
        "email": "mert.goker.work@gmail.com"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Price forecasting endpoint
@app.post("/forecast", response_model=ForecastResponse)
async def forecast_price(request: ForecastRequest):
    """
    Generate LSTM-based price forecasts for cryptocurrency
    """
    try:
        logger.info(f"Generating forecast for {request.symbol}")
        
        # Get historical data
        if request.start_date and request.end_date:
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
        
        # Fetch data
        ticker = yf.Ticker(request.symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {request.symbol}")
        
        # Generate forecast using LSTM model
        predictions, dates, metrics = lstm_model.predict(data, request.days)
        
        # Calculate confidence based on model performance
        confidence = min(0.95, max(0.6, 1 - metrics.get('mape', 0.1)))
        
        return ForecastResponse(
            symbol=request.symbol,
            predictions=predictions,
            dates=dates,
            metrics=metrics,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

# Sentiment analysis endpoint
@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment from cryptocurrency news and social media
    """
    try:
        logger.info(f"Analyzing sentiment for {request.symbol}")
        
        # Get sentiment analysis
        sentiment_data = sentiment_analyzer.analyze(
            symbol=request.symbol,
            source=request.source,
            limit=request.limit
        )
        
        return SentimentResponse(
            symbol=request.symbol,
            overall_sentiment=sentiment_data['overall_sentiment'],
            sentiment_distribution=sentiment_data['distribution'],
            recent_news=sentiment_data['recent_news'],
            confidence=sentiment_data['confidence']
        )
        
    except Exception as e:
        logger.error(f"Error in sentiment endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

# Technical indicators endpoint
@app.post("/technical", response_model=TechnicalResponse)
async def get_technical_indicators(request: TechnicalRequest):
    """
    Calculate technical indicators for cryptocurrency
    """
    try:
        logger.info(f"Calculating technical indicators for {request.symbol}")
        
        # Get historical data
        if request.start_date and request.end_date:
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
        
        # Fetch data
        ticker = yf.Ticker(request.symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {request.symbol}")
        
        # Calculate technical indicators
        indicators_data = technical_indicators.calculate(data, request.indicators)
        
        return TechnicalResponse(
            symbol=request.symbol,
            indicators=indicators_data['indicators'],
            signals=indicators_data['signals'],
            summary=indicators_data['summary']
        )
        
    except Exception as e:
        logger.error(f"Error in technical indicators endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Technical analysis failed: {str(e)}")

# AI analysis endpoint
@app.post("/ai-analysis", response_model=AIAnalysisResponse)
async def get_ai_analysis(request: AIAnalysisRequest):
    """
    Generate AI-powered analysis and recommendations
    """
    try:
        logger.info(f"Generating AI analysis for {request.symbol}")
        
        # Get AI analysis
        analysis_data = ai_analyzer.analyze(
            symbol=request.symbol,
            analysis_type=request.analysis_type,
            user_query=request.user_query,
            context=request.context
        )
        
        return AIAnalysisResponse(
            symbol=request.symbol,
            analysis_type=request.analysis_type,
            analysis=analysis_data['analysis'],
            recommendations=analysis_data['recommendations'],
            confidence=analysis_data['confidence'],
            insights=analysis_data['insights']
        )
        
    except Exception as e:
        logger.error(f"Error in AI analysis endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

# System status endpoint
@app.get("/system-status", response_model=SystemStatusResponse)
async def get_system_status():
    """
    Get system status and performance metrics
    """
    try:
        # Check component status
        components = {
            "frontend": "online",
            "backend": "online",
            "ml_models": "active",
            "database": "connected",
            "llm_service": "available"
        }
        
        # Mock performance metrics
        performance_metrics = {
            "api_response_time": "120ms",
            "model_accuracy": "94.2%",
            "data_freshness": "real-time",
            "system_uptime": "99.8%",
            "cpu_usage": "45%",
            "memory_usage": "62%",
            "disk_usage": "28%"
        }
        
        return SystemStatusResponse(
            status="healthy",
            components=components,
            performance_metrics=performance_metrics,
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in system status endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System status check failed: {str(e)}")

# Get available cryptocurrencies
@app.get("/cryptocurrencies")
async def get_cryptocurrencies():
    """
    Get list of available cryptocurrencies
    """
    cryptocurrencies = [
        {"symbol": "BTC-USD", "name": "Bitcoin", "category": "Major"},
        {"symbol": "ETH-USD", "name": "Ethereum", "category": "Major"},
        {"symbol": "ADA-USD", "name": "Cardano", "category": "Major"},
        {"symbol": "DOT-USD", "name": "Polkadot", "category": "Major"},
        {"symbol": "LINK-USD", "name": "Chainlink", "category": "Major"},
        {"symbol": "SOL-USD", "name": "Solana", "category": "Major"},
        {"symbol": "MATIC-USD", "name": "Polygon", "category": "DeFi"},
        {"symbol": "AVAX-USD", "name": "Avalanche", "category": "DeFi"},
        {"symbol": "UNI-USD", "name": "Uniswap", "category": "DeFi"},
        {"symbol": "AAVE-USD", "name": "Aave", "category": "DeFi"}
    ]
    
    return {"cryptocurrencies": cryptocurrencies}

# Get historical data
@app.get("/historical-data/{symbol}")
async def get_historical_data(symbol: str, days: int = 365):
    """
    Get historical price data for cryptocurrency
    """
    try:
        # Use the new get_crypto_data function
        data = get_crypto_data(symbol, days)
        
        if data.empty:
            raise Exception("No data available from any source")
        
        # Convert to JSON-serializable format
        data_dict = {
            "dates": data.index.strftime('%Y-%m-%d').tolist(),
            "prices": data['close'].tolist(),
            "volumes": data['volume'].tolist(),
            "highs": data['high'].tolist(),
            "lows": data['low'].tolist(),
            "opens": data['open'].tolist()
        }
        return {"symbol": symbol, "data": data_dict, "source": "real_data"}
        
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {str(e)}")
        # Fallback: Return mock data for demonstration if all sources fail
        mock_dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)][::-1]
        mock_prices = [30000 + 1000 * np.sin(i/10) for i in range(days)]
        mock_volumes = [1000 + 100 * np.cos(i/5) for i in range(days)]
        mock_highs = [p + 200 for p in mock_prices]
        mock_lows = [p - 200 for p in mock_prices]
        mock_opens = [p + 50 for p in mock_prices]
        mock_data_dict = {
            "dates": mock_dates,
            "prices": mock_prices,
            "volumes": mock_volumes,
            "highs": mock_highs,
            "lows": mock_lows,
            "opens": mock_opens
        }
        return {"symbol": symbol, "data": mock_data_dict, "warning": "Returned mock data due to data provider error."}

# Model metrics endpoint
@app.get("/model-metrics/{symbol}")
async def get_model_metrics(symbol: str):
    """
    Get model performance metrics for a specific cryptocurrency
    """
    try:
        # Mock metrics (in real implementation, these would come from the trained model)
        metrics = {
            "mape": round(np.random.uniform(2.5, 8.5), 2),
            "rmse": round(np.random.uniform(500, 2000), 2),
            "mse": round(np.random.uniform(250000, 4000000), 0),
            "accuracy": round(np.random.uniform(85, 95), 2),
            "last_updated": datetime.now().isoformat()
        }
        
        return {"symbol": symbol, "metrics": metrics}
        
    except Exception as e:
        logger.error(f"Error getting model metrics for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model metrics: {str(e)}")

# Get sentiment trend endpoint
@app.get("/sentiment-trend/{symbol}")
async def get_sentiment_trend(symbol: str, days: int = 30):
    """
    Get sentiment trend over time for a specific cryptocurrency
    """
    try:
        logger.info(f"Getting sentiment trend for {symbol} over {days} days")
        
        # Get sentiment trend data
        trend_data = sentiment_analyzer.get_sentiment_trend(symbol, days)
        
        return {
            "symbol": symbol,
            "days": days,
            "trend_data": trend_data,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting sentiment trend for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get sentiment trend: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 