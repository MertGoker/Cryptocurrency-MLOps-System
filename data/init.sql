-- Cryptocurrency MLOps System Database Initialization

-- Create tables for storing cryptocurrency data and model results

-- Cryptocurrency price data table
CREATE TABLE IF NOT EXISTS crypto_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(20, 8),
    high_price DECIMAL(20, 8),
    low_price DECIMAL(20, 8),
    close_price DECIMAL(20, 8),
    volume DECIMAL(20, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

-- Model predictions table
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    prediction_date DATE NOT NULL,
    predicted_price DECIMAL(20, 8),
    confidence DECIMAL(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    mape DECIMAL(10, 4),
    rmse DECIMAL(20, 8),
    mse DECIMAL(20, 8),
    accuracy DECIMAL(5, 4),
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sentiment analysis results table
CREATE TABLE IF NOT EXISTS sentiment_analysis (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    source VARCHAR(100),
    sentiment_score DECIMAL(5, 4),
    sentiment_label VARCHAR(20),
    text_content TEXT,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Technical indicators table
CREATE TABLE IF NOT EXISTS technical_indicators (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    rsi DECIMAL(10, 4),
    macd DECIMAL(20, 8),
    macd_signal DECIMAL(20, 8),
    bollinger_upper DECIMAL(20, 8),
    bollinger_middle DECIMAL(20, 8),
    bollinger_lower DECIMAL(20, 8),
    sma_20 DECIMAL(20, 8),
    sma_50 DECIMAL(20, 8),
    sma_200 DECIMAL(20, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

-- AI analysis results table
CREATE TABLE IF NOT EXISTS ai_analysis (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    analysis_type VARCHAR(50) NOT NULL,
    analysis_text TEXT,
    recommendations TEXT[],
    confidence DECIMAL(5, 4),
    insights TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System logs table
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(50) NOT NULL,
    log_level VARCHAR(20) NOT NULL,
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_crypto_prices_symbol_date ON crypto_prices(symbol, date);
CREATE INDEX IF NOT EXISTS idx_model_predictions_symbol ON model_predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_sentiment_analysis_symbol ON sentiment_analysis(symbol);
CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_date ON technical_indicators(symbol, date);
CREATE INDEX IF NOT EXISTS idx_ai_analysis_symbol ON ai_analysis(symbol);
CREATE INDEX IF NOT EXISTS idx_system_logs_created_at ON system_logs(created_at);

-- Insert sample data for testing
INSERT INTO crypto_prices (symbol, date, open_price, high_price, low_price, close_price, volume) VALUES
('BTC-USD', '2024-01-01', 45000.00, 45500.00, 44800.00, 45200.00, 1000000.00),
('BTC-USD', '2024-01-02', 45200.00, 45800.00, 45000.00, 45600.00, 1200000.00),
('ETH-USD', '2024-01-01', 2500.00, 2550.00, 2480.00, 2520.00, 800000.00),
('ETH-USD', '2024-01-02', 2520.00, 2580.00, 2500.00, 2560.00, 900000.00)
ON CONFLICT (symbol, date) DO NOTHING;

-- Create views for common queries
CREATE OR REPLACE VIEW latest_prices AS
SELECT DISTINCT ON (symbol) 
    symbol, 
    date, 
    close_price,
    volume
FROM crypto_prices 
ORDER BY symbol, date DESC;

CREATE OR REPLACE VIEW model_performance_summary AS
SELECT 
    symbol,
    model_type,
    AVG(mape) as avg_mape,
    AVG(rmse) as avg_rmse,
    AVG(accuracy) as avg_accuracy,
    COUNT(*) as model_runs
FROM model_metrics 
GROUP BY symbol, model_type;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlops_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlops_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO mlops_user; 