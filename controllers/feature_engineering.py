
import pandas as pd
import numpy as np
import ta
import logging

logger = logging.getLogger(__name__)

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the given DataFrame with proper bounds checking"""
    # Add basic price data if not present
    if 'High' not in df.columns:
        df['High'] = df['Close']
    if 'Low' not in df.columns:
        df['Low'] = df['Close']
    if 'Open' not in df.columns:
        df['Open'] = df['Close'].shift(1)
    if 'Volume' not in df.columns:
        df['Volume'] = 1000000  # Default volume

    # Ensure all prices are positive
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col].abs()
    
    # Ensure High >= Low and High/Low contain Close
    df['High'] = df[['High', 'Close']].max(axis=1)
    df['Low'] = df[['Low', 'Close']].min(axis=1)
    
    # Default windows for indicators
    sma_window = 20
    ema_span = 12
    bb_window = 20
    rsi_window = 14
    momentum_window = 10
    volatility_window = 20
    epsilon = 1e-10  # Small value to prevent division by zero

    # Technical Indicators with proper error handling
    try:
        # SMA
        df['SMA_20'] = df['Close'].rolling(window=sma_window).mean()
        
        # EMA
        df['EMA_20'] = df['Close'].ewm(span=ema_span, adjust=False).mean()
        
        # RSI with proper bounds (0-100)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_window).mean()
        rs = gain / (loss + epsilon)
        rsi = 100 - (100 / (1 + rs))
        # Ensure RSI is between 0 and 100
        df['RSI_14'] = np.clip(rsi, 0, 100)
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        bb_middle = df['Close'].rolling(window=bb_window).mean()
        bb_std = df['Close'].rolling(window=bb_window).std()
        df['Bollinger_middle'] = bb_middle
        df['Bollinger_Upper'] = bb_middle + 2 * bb_std
        df['Bollinger_Lower'] = bb_middle - 2 * bb_std
        
        # Momentum
        df['Momentum'] = df['Close'].diff(momentum_window)
        
        # Volatility (always positive)
        df['Volatility'] = df['Close'].rolling(window=volatility_window).std().abs()
        
        # Lag1
        df['Lag1'] = df['Close'].shift(1)
        
        # Sentiment_Up (placeholder - should be updated with actual sentiment data)
        df['Sentiment_Up'] = 0.5  # Default neutral sentiment
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        # Set default values if calculation fails
        for col in ['SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 
                   'Bollinger_middle', 'Bollinger_Upper', 'Bollinger_Lower', 'Momentum', 
                   'Volatility', 'Lag1', 'Sentiment_Up']:
            if col not in df.columns:
                if col == 'RSI_14':
                    df[col] = 50.0  # Neutral RSI
                elif col == 'Sentiment_Up':
                    df[col] = 0.5   # Neutral sentiment
                else:
                    df[col] = df['Close']  # Default to close price
    
    # Handle missing values
    df = df.ffill().bfill()
    
    # Replace infinite values with NaN and then fill them
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    
    # Final bounds checking
    if 'RSI_14' in df.columns:
        df['RSI_14'] = np.clip(df['RSI_14'], 0, 100)
    
    # Ensure all numeric columns are finite
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    return df
