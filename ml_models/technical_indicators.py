import numpy as np
import pandas as pd
import ta
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Technical indicators calculator for cryptocurrency analysis
    """
    
    def __init__(self):
        self.available_indicators = [
            'rsi', 'macd', 'bollinger_bands', 'sma', 'ema', 
            'stochastic', 'williams_r', 'cci', 'adx', 'obv'
        ]
    
    def calculate(self, data: pd.DataFrame, indicators: List[str]) -> Dict[str, Any]:
        """
        Calculate technical indicators for the given data
        """
        try:
            logger.info(f"Calculating technical indicators: {indicators}")
            
            result = {
                'indicators': {},
                'signals': {},
                'summary': ''
            }
            
            # Calculate each requested indicator
            for indicator in indicators:
                if indicator in self.available_indicators:
                    indicator_data = self.calculate_indicator(data, indicator)
                    result['indicators'][indicator] = indicator_data
            
            # Generate trading signals
            result['signals'] = self.generate_signals(data, result['indicators'])
            
            # Generate summary
            result['summary'] = self.generate_summary(result['signals'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return self.get_default_response()
    
    def calculate_indicator(self, data: pd.DataFrame, indicator: str) -> Dict[str, Any]:
        """
        Calculate a specific technical indicator
        """
        try:
            if indicator == 'rsi':
                return self.calculate_rsi(data)
            elif indicator == 'macd':
                return self.calculate_macd(data)
            elif indicator == 'bollinger_bands':
                return self.calculate_bollinger_bands(data)
            elif indicator == 'sma':
                return self.calculate_sma(data)
            elif indicator == 'ema':
                return self.calculate_ema(data)
            elif indicator == 'stochastic':
                return self.calculate_stochastic(data)
            elif indicator == 'williams_r':
                return self.calculate_williams_r(data)
            elif indicator == 'cci':
                return self.calculate_cci(data)
            elif indicator == 'adx':
                return self.calculate_adx(data)
            elif indicator == 'obv':
                return self.calculate_obv(data)
            else:
                logger.warning(f"Unknown indicator: {indicator}")
                return {}
                
        except Exception as e:
            logger.error(f"Error calculating {indicator}: {str(e)}")
            return {}
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """
        Calculate Relative Strength Index (RSI)
        """
        try:
            rsi = ta.momentum.RSIIndicator(data['Close'], window=period)
            rsi_values = rsi.rsi()
            
            return {
                'values': rsi_values.tolist(),
                'current': float(rsi_values.iloc[-1]) if not pd.isna(rsi_values.iloc[-1]) else 50.0,
                'period': period,
                'overbought': 70,
                'oversold': 30
            }
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return {}
    
    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Any]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        """
        try:
            macd_indicator = ta.trend.MACD(data['Close'], window_fast=fast, window_slow=slow, window_sign=signal)
            macd_line = macd_indicator.macd()
            signal_line = macd_indicator.macd_signal()
            histogram = macd_indicator.macd_diff()
            
            return {
                'macd_line': macd_line.tolist(),
                'signal_line': signal_line.tolist(),
                'histogram': histogram.tolist(),
                'current_macd': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
                'current_signal': float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
                'current_histogram': float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0,
                'fast_period': fast,
                'slow_period': slow,
                'signal_period': signal
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return {}
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2) -> Dict[str, Any]:
        """
        Calculate Bollinger Bands
        """
        try:
            bb_indicator = ta.volatility.BollingerBands(data['Close'], window=period, window_dev=std_dev)
            upper_band = bb_indicator.bollinger_hband()
            middle_band = bb_indicator.bollinger_mavg()
            lower_band = bb_indicator.bollinger_lband()
            
            current_price = data['Close'].iloc[-1]
            
            return {
                'upper_band': upper_band.tolist(),
                'middle_band': middle_band.tolist(),
                'lower_band': lower_band.tolist(),
                'current_price': float(current_price),
                'current_upper': float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else 0.0,
                'current_middle': float(middle_band.iloc[-1]) if not pd.isna(middle_band.iloc[-1]) else 0.0,
                'current_lower': float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else 0.0,
                'period': period,
                'std_dev': std_dev
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return {}
    
    def calculate_sma(self, data: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> Dict[str, Any]:
        """
        Calculate Simple Moving Averages
        """
        try:
            sma_data = {}
            for period in periods:
                sma = ta.trend.SMAIndicator(data['Close'], window=period)
                sma_values = sma.sma_indicator()
                sma_data[f'sma_{period}'] = {
                    'values': sma_values.tolist(),
                    'current': float(sma_values.iloc[-1]) if not pd.isna(sma_values.iloc[-1]) else 0.0,
                    'period': period
                }
            
            return sma_data
        except Exception as e:
            logger.error(f"Error calculating SMA: {str(e)}")
            return {}
    
    def calculate_ema(self, data: pd.DataFrame, periods: List[int] = [12, 26]) -> Dict[str, Any]:
        """
        Calculate Exponential Moving Averages
        """
        try:
            ema_data = {}
            for period in periods:
                ema = ta.trend.EMAIndicator(data['Close'], window=period)
                ema_values = ema.ema_indicator()
                ema_data[f'ema_{period}'] = {
                    'values': ema_values.tolist(),
                    'current': float(ema_values.iloc[-1]) if not pd.isna(ema_values.iloc[-1]) else 0.0,
                    'period': period
                }
            
            return ema_data
        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            return {}
    
    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, Any]:
        """
        Calculate Stochastic Oscillator
        """
        try:
            stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], 
                                                   window=k_period, smooth_window=d_period)
            k_line = stoch.stoch()
            d_line = stoch.stoch_signal()
            
            return {
                'k_line': k_line.tolist(),
                'd_line': d_line.tolist(),
                'current_k': float(k_line.iloc[-1]) if not pd.isna(k_line.iloc[-1]) else 50.0,
                'current_d': float(d_line.iloc[-1]) if not pd.isna(d_line.iloc[-1]) else 50.0,
                'k_period': k_period,
                'd_period': d_period,
                'overbought': 80,
                'oversold': 20
            }
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            return {}
    
    def calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """
        Calculate Williams %R
        """
        try:
            williams_r = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close'], window=period)
            wr_values = williams_r.williams_r()
            
            return {
                'values': wr_values.tolist(),
                'current': float(wr_values.iloc[-1]) if not pd.isna(wr_values.iloc[-1]) else -50.0,
                'period': period,
                'overbought': -20,
                'oversold': -80
            }
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {str(e)}")
            return {}
    
    def calculate_cci(self, data: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
        """
        Calculate Commodity Channel Index (CCI)
        """
        try:
            cci = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close'], window=period)
            cci_values = cci.cci()
            
            return {
                'values': cci_values.tolist(),
                'current': float(cci_values.iloc[-1]) if not pd.isna(cci_values.iloc[-1]) else 0.0,
                'period': period,
                'overbought': 100,
                'oversold': -100
            }
        except Exception as e:
            logger.error(f"Error calculating CCI: {str(e)}")
            return {}
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """
        Calculate Average Directional Index (ADX)
        """
        try:
            adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=period)
            adx_values = adx.adx()
            plus_di = adx.adx_pos()
            minus_di = adx.adx_neg()
            
            return {
                'adx_values': adx_values.tolist(),
                'plus_di': plus_di.tolist(),
                'minus_di': minus_di.tolist(),
                'current_adx': float(adx_values.iloc[-1]) if not pd.isna(adx_values.iloc[-1]) else 0.0,
                'current_plus_di': float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0,
                'current_minus_di': float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0,
                'period': period,
                'strong_trend': 25
            }
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            return {}
    
    def calculate_obv(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate On-Balance Volume (OBV)
        """
        try:
            obv = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume'])
            obv_values = obv.on_balance_volume()
            
            return {
                'values': obv_values.tolist(),
                'current': float(obv_values.iloc[-1]) if not pd.isna(obv_values.iloc[-1]) else 0.0
            }
        except Exception as e:
            logger.error(f"Error calculating OBV: {str(e)}")
            return {}
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate trading signals based on technical indicators
        """
        try:
            signals = {}
            current_price = data['Close'].iloc[-1]
            
            # RSI signals
            if 'rsi' in indicators:
                rsi_current = indicators['rsi']['current']
                if rsi_current > 70:
                    signals['rsi'] = '🔴 Overbought'
                elif rsi_current < 30:
                    signals['rsi'] = '🟢 Oversold'
                else:
                    signals['rsi'] = '🟡 Neutral'
            
            # MACD signals
            if 'macd' in indicators:
                macd_current = indicators['macd']['current_macd']
                signal_current = indicators['macd']['current_signal']
                if macd_current > signal_current:
                    signals['macd'] = '🟢 Bullish'
                else:
                    signals['macd'] = '🔴 Bearish'
            
            # Bollinger Bands signals
            if 'bollinger_bands' in indicators:
                bb = indicators['bollinger_bands']
                if current_price > bb['current_upper']:
                    signals['bollinger_bands'] = '🔴 Overbought'
                elif current_price < bb['current_lower']:
                    signals['bollinger_bands'] = '🟢 Oversold'
                else:
                    signals['bollinger_bands'] = '🟡 Normal'
            
            # SMA signals
            if 'sma' in indicators:
                sma_20 = indicators['sma'].get('sma_20', {}).get('current', 0)
                sma_50 = indicators['sma'].get('sma_50', {}).get('current', 0)
                if current_price > sma_20 > sma_50:
                    signals['sma'] = '🟢 Bullish'
                elif current_price < sma_20 < sma_50:
                    signals['sma'] = '🔴 Bearish'
                else:
                    signals['sma'] = '🟡 Neutral'
            
            # Stochastic signals
            if 'stochastic' in indicators:
                stoch = indicators['stochastic']
                k_current = stoch['current_k']
                d_current = stoch['current_d']
                if k_current > 80 and d_current > 80:
                    signals['stochastic'] = '🔴 Overbought'
                elif k_current < 20 and d_current < 20:
                    signals['stochastic'] = '🟢 Oversold'
                else:
                    signals['stochastic'] = '🟡 Neutral'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return {}
    
    def generate_summary(self, signals: Dict[str, str]) -> str:
        """
        Generate a summary of technical analysis signals
        """
        try:
            if not signals:
                return "No technical signals available."
            
            bullish_count = sum(1 for signal in signals.values() if '🟢' in signal)
            bearish_count = sum(1 for signal in signals.values() if '🔴' in signal)
            neutral_count = sum(1 for signal in signals.values() if '🟡' in signal)
            
            total_signals = len(signals)
            
            if bullish_count > bearish_count and bullish_count > neutral_count:
                overall_sentiment = "Bullish"
                emoji = "🟢"
            elif bearish_count > bullish_count and bearish_count > neutral_count:
                overall_sentiment = "Bearish"
                emoji = "🔴"
            else:
                overall_sentiment = "Neutral"
                emoji = "🟡"
            
            summary = f"{emoji} Overall Technical Sentiment: {overall_sentiment}\n"
            summary += f"Bullish signals: {bullish_count}/{total_signals}\n"
            summary += f"Bearish signals: {bearish_count}/{total_signals}\n"
            summary += f"Neutral signals: {neutral_count}/{total_signals}"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Unable to generate technical analysis summary."
    
    def get_default_response(self) -> Dict[str, Any]:
        """
        Return default response when calculation fails
        """
        return {
            'indicators': {},
            'signals': {},
            'summary': 'Technical analysis calculation failed.'
        } 