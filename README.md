# Cryptocurrency MLOps System

A comprehensive MLOps system for cryptocurrency analysis featuring LSTM price forecasting, sentiment analysis, technical indicators, and LLM-powered recommendations.

## 🚀 Features

- **LSTM-based Cryptocurrency Price Forecasting** with MAPE, RMSE, and MSE metrics
- **Sentiment Analysis** from recent crypto news using free HuggingFace models
- **Technical Indicators & Performance Dashboard**
- **Free LLM-powered Analysis and Recommendations** using advanced free models
- **Jenkins CI/CD Pipeline**
- **Optional LangSmith Integration** for LLM tracing and evaluation
- **Docker Containerization**
- **FastAPI Backend** with **Streamlit Frontend**

## 🏗️ Architecture

```
├── frontend/          # Streamlit application
├── backend/           # FastAPI server
├── ml_models/         # LSTM models and training
├── data/              # Data storage and processing
├── docker/            # Docker configurations
├── jenkins/           # CI/CD pipeline
└── docs/              # Documentation
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **ML**: TensorFlow/Keras (LSTM), Transformers (Sentiment)
- **LLM**: Free models (Llama 3.2 compatible) with RAG pipeline
- **Monitoring**: Optional LangSmith
- **Containerization**: Docker
- **CI/CD**: Jenkins
- **Data**: Pandas, NumPy, yfinance

## 📊 Model Metrics

The LSTM model is evaluated using:
- **MAPE** (Mean Absolute Percentage Error)
- **RMSE** (Root Mean Square Error)  
- **MSE** (Mean Square Error)

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Cryptocurrency-MLOps-System
   ```

2. **Run with Docker**
   ```bash
   docker-compose up --build
   ```

3. **Access the application**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 👨‍💻 Developer

**Mert Göker** - Data Scientist

- GitHub: [@MertGoker](https://github.com/MertGoker)
- LinkedIn: [Mert Göker](https://www.linkedin.com/in/mert-goker-bb4bb91b6/)
- Email: mert.goker.work@gmail.com

## 📝 License

This project is licensed under the MIT License. 