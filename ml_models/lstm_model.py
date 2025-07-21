import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class LSTMModel:
    """
    LSTM model for cryptocurrency price forecasting with comprehensive evaluation metrics
    """
    
    def __init__(self, sequence_length=60, units=50, dropout=0.2, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.is_trained = False
        
    def prepare_data(self, data, target_column='Close'):
        """
        Prepare data for LSTM model training
        """
        try:
            # Use only the target column
            dataset = data[target_column].values.reshape(-1, 1)
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(dataset)
            
            # Create sequences for training
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            return X, y, scaled_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        """
        try:
            model = Sequential([
                LSTM(units=self.units, return_sequences=True, input_shape=input_shape),
                Dropout(self.dropout),
                LSTM(units=self.units, return_sequences=True),
                Dropout(self.dropout),
                LSTM(units=self.units, return_sequences=False),
                Dropout(self.dropout),
                Dense(units=25),
                Dense(units=1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mean_squared_error'
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def train(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the LSTM model
        """
        try:
            logger.info("Starting LSTM model training...")
            
            # Prepare data
            X, y, scaled_data = self.prepare_data(data)
            
            # Split data into training and validation sets
            train_size = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Build model
            self.model = self.build_model((X.shape[1], 1))
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                verbose=1
            )
            
            self.is_trained = True
            logger.info("LSTM model training completed successfully")
            
            return history
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, data, days=30, target_column='Close'):
        """
        Generate predictions and calculate metrics
        """
        try:
            if not self.is_trained:
                logger.info("Model not trained. Training with provided data...")
                self.train(data)
            
            # Prepare data
            X, y, scaled_data = self.prepare_data(data, target_column)
            
            # Make predictions on training data for metrics calculation
            train_predictions = self.model.predict(X)
            train_predictions = self.scaler.inverse_transform(train_predictions)
            actual_values = self.scaler.inverse_transform(y.reshape(-1, 1))
            
            # Calculate metrics
            metrics = self.calculate_metrics(actual_values, train_predictions)
            
            # Generate future predictions
            future_predictions = self.generate_future_predictions(scaled_data, days)
            
            # Generate dates for predictions
            last_date = data.index[-1]
            prediction_dates = [last_date + timedelta(days=i+1) for i in range(days)]
            date_strings = [date.strftime('%Y-%m-%d') for date in prediction_dates]
            
            return future_predictions, date_strings, metrics
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def generate_future_predictions(self, scaled_data, days):
        """
        Generate future predictions using the trained model
        """
        try:
            # Get the last sequence for prediction
            last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days):
                # Predict next value
                next_pred = self.model.predict(current_sequence, verbose=0)
                predictions.append(next_pred[0, 0])
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[0, -1, 0] = next_pred[0, 0]
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            
            return predictions.flatten().tolist()
            
        except Exception as e:
            logger.error(f"Error generating future predictions: {str(e)}")
            raise
    
    def calculate_metrics(self, actual, predicted):
        """
        Calculate MAPE, RMSE, and MSE metrics
        """
        try:
            # Flatten arrays
            actual_flat = actual.flatten()
            predicted_flat = predicted.flatten()
            
            # Calculate metrics
            mse = mean_squared_error(actual_flat, predicted_flat)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE (handle division by zero)
            mape = mean_absolute_percentage_error(actual_flat, predicted_flat) * 100
            
            # Additional metrics
            mae = np.mean(np.abs(actual_flat - predicted_flat))
            
            return {
                "mape": mape,
                "rmse": rmse,
                "mse": mse,
                "mae": mae
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def save_model(self, filepath):
        """
        Save the trained model to a file
        """
        try:
            if self.model is not None:
                self.model.save(filepath)
                logger.info(f"Model saved to {filepath}")
            else:
                logger.warning("No model to save.")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath):
        """
        Load a trained model from a file
        """
        try:
            self.model = tf.keras.models.load_model(filepath)
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_model_summary(self):
        """
        Get a summary of the model architecture
        """
        if self.model is not None:
            self.model.summary()
        else:
            logger.warning("No model to summarize.") 