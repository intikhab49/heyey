#!/usr/bin/env python3
"""
Enhanced Data Quality and Feature Engineering Module
Addresses the high correlation issues and improves model accuracy
"""

import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class EnhancedFeatureProcessor:
    """Enhanced feature processor to handle correlation and improve model accuracy"""
    
    def __init__(self, correlation_threshold=0.95, variance_threshold=0.01):
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.feature_selector = None
        self.pca = None
        self.removed_features = []
        # Protect essential OHLCV and model features from removal
        self.protected_columns = {
            'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 
            'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Bollinger_middle', 
            'Bollinger_Upper', 'Bollinger_Lower', 'ATR', 'OBV', 'VWAP', 'CCI', 
            'Stoch_%K', 'Stoch_%D', 'MFI', 'Momentum', 'Volatility', 'Lag1', 
            'Sentiment_Up', 'ADX'
        }
        
    def remove_highly_correlated_features(self, df):
        """Remove features with high correlation, protecting essential columns"""
        corr_matrix = df.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with high correlation, but protect essential columns
        to_drop = []
        for column in upper_triangle.columns:
            if column in self.protected_columns:
                continue  # Skip protected columns
            if any(upper_triangle[column] > self.correlation_threshold):
                to_drop.append(column)
        
        self.removed_features.extend(to_drop)
        logger.info(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
        
        return df.drop(columns=to_drop)
    
    def remove_low_variance_features(self, df):
        """Remove features with very low variance, protecting essential columns"""
        low_variance = df.var() < self.variance_threshold
        low_var_features = [col for col in low_variance[low_variance].index 
                           if col not in self.protected_columns]
        
        if low_var_features:
            self.removed_features.extend(low_var_features)
            logger.info(f"Removing {len(low_var_features)} low variance features: {low_var_features}")
            df = df.drop(columns=low_var_features)
        
        return df
    
    def add_enhanced_technical_indicators(self, df):
        """Add enhanced technical indicators with better signal quality"""
        try:
            # Enhanced momentum indicators
            df['Price_Change_Pct'] = df['Close'].pct_change()
            df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
            df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open']
            
            # Enhanced volatility indicators
            df['Price_Volatility_5'] = df['Close'].rolling(5).std() / df['Close'].rolling(5).mean()
            df['Price_Volatility_10'] = df['Close'].rolling(10).std() / df['Close'].rolling(10).mean()
            
            # Volume-price relationship
            df['Volume_Price_Trend'] = (df['Volume'] * df['Close']).rolling(10).mean()
            df['Volume_SMA_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            # Support and resistance levels
            df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
            df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
            df['Body_Size'] = abs(df['Close'] - df['Open'])
            
            # Trend strength indicators
            df['Trend_Strength'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
            
            # Money flow indicators (enhanced)
            df['Money_Flow_Multiplier'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            df['Money_Flow_Volume'] = df['Money_Flow_Multiplier'] * df['Volume']
            df['Money_Flow_Index'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Price position indicators
            df['Price_Position'] = (df['Close'] - df['Low'].rolling(20).min()) / (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
            
            # Rate of change indicators
            df['ROC_5'] = ta.momentum.roc(df['Close'], window=5)
            df['ROC_10'] = ta.momentum.roc(df['Close'], window=10)
            
            # Enhanced moving average indicators
            df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
            df['MACD_Enhanced'] = df['EMA_12'] - df['EMA_26']
            
            # Bollinger Band position
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_Position'] = (df['Close'] - bollinger.bollinger_lband()) / (bollinger.bollinger_hband() - bollinger.bollinger_lband())
            
            # RSI variations
            df['RSI_9'] = ta.momentum.rsi(df['Close'], window=9)
            df['RSI_21'] = ta.momentum.rsi(df['Close'], window=21)
            
            # Stochastic variations
            stoch_fast = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=5)
            df['Stoch_Fast'] = stoch_fast.stoch()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding enhanced technical indicators: {str(e)}")
            return df
    
    def clean_and_scale_features(self, df, target_column='Close'):
        """Clean, scale and prepare features for model training"""
        try:
            # Separate target and features
            if target_column in df.columns:
                target = df[target_column].copy()
                feature_df = df.drop(columns=[target_column])
            else:
                target = df.iloc[:, -1].copy()  # Assume last column is target
                feature_df = df.iloc[:, :-1].copy()
            
            # Handle infinite and NaN values
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
            feature_df = feature_df.ffill().bfill()
            
            # Remove constant features, protecting essential columns
            constant_features = [col for col in feature_df.columns 
                               if feature_df[col].nunique() <= 1 and col not in self.protected_columns]
            if constant_features:
                logger.info(f"Removing constant features: {constant_features}")
                feature_df = feature_df.drop(columns=constant_features)
            
            # Remove low variance features
            feature_df = self.remove_low_variance_features(feature_df)
            
            # Remove highly correlated features
            feature_df = self.remove_highly_correlated_features(feature_df)
            
            # Scale features using RobustScaler (better for outliers)
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(feature_df)
            scaled_df = pd.DataFrame(scaled_features, columns=feature_df.columns, index=feature_df.index)
            
            # Add target back
            scaled_df[target_column] = target
            
            logger.info(f"Feature processing complete. Final features: {len(feature_df.columns)}")
            logger.info(f"Removed features due to correlation/variance: {len(self.removed_features)}")
            
            return scaled_df, scaler
            
        except Exception as e:
            logger.error(f"Error in clean_and_scale_features: {str(e)}")
            return df, None
    
    def select_best_features(self, X, y, k=15):
        """Select k best features using statistical tests"""
        try:
            # Ensure we don't select more features than available
            k = min(k, X.shape[1])
            
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            
            selected_features = X.columns[self.feature_selector.get_support()]
            logger.info(f"Selected {len(selected_features)} best features: {list(selected_features)}")
            
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            return X

def get_enhanced_feature_list():
    """Get enhanced feature list for the model"""
    return [
        # Core OHLCV
        'Open', 'High', 'Low', 'Close', 'Volume',
        
        # Enhanced price action
        'Price_Change_Pct', 'High_Low_Pct', 'Open_Close_Pct',
        
        # Enhanced volatility
        'Price_Volatility_5', 'Price_Volatility_10',
        
        # Volume analysis
        'Volume_Price_Trend', 'Volume_SMA_Ratio',
        
        # Candlestick patterns
        'Upper_Shadow', 'Lower_Shadow', 'Body_Size',
        
        # Trend and momentum
        'Trend_Strength', 'ROC_5', 'ROC_10',
        
        # Money flow
        'Money_Flow_Index', 'Money_Flow_Volume',
        
        # Position indicators
        'Price_Position', 'BB_Position',
        
        # Enhanced moving averages
        'EMA_12', 'EMA_26', 'MACD_Enhanced',
        
        # RSI variations
        'RSI_9', 'RSI_21',
        
        # Stochastic
        'Stoch_Fast'
    ]

# Enhanced configuration for better accuracy
ENHANCED_TIMEFRAME_CONFIG = {
    '30m': {
        'sequence_length': 48,  # 24 hours of 30m candles
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'patience': 10,
        'feature_selection_k': 15
    },
    '1h': {
        'sequence_length': 24,  # 24 hours
        'hidden_size': 96,
        'num_layers': 3,
        'dropout': 0.25,
        'learning_rate': 0.0008,
        'batch_size': 16,
        'epochs': 75,
        'patience': 15,
        'feature_selection_k': 18
    },
    '4h': {
        'sequence_length': 18,  # 3 days
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.3,
        'learning_rate': 0.0005,
        'batch_size': 8,
        'epochs': 100,
        'patience': 20,
        'feature_selection_k': 20
    },
    '24h': {
        'sequence_length': 14,  # 2 weeks
        'hidden_size': 96,
        'num_layers': 2,
        'dropout': 0.15,
        'learning_rate': 0.0003,
        'batch_size': 4,
        'epochs': 150,
        'patience': 30,
        'feature_selection_k': 12
    }
}

if __name__ == "__main__":
    # Test the enhanced feature processor
    print("Enhanced Feature Processor ready for integration")
    print(f"Available enhanced features: {len(get_enhanced_feature_list())}")
    for timeframe, config in ENHANCED_TIMEFRAME_CONFIG.items():
        print(f"{timeframe}: {config['feature_selection_k']} features, {config['sequence_length']} sequence length")
