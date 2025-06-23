import os
import shutil
import json
import joblib
from datetime import datetime
import pytz
import logging
from typing import Dict, Optional, Tuple, Any
import torch
import hashlib
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

class ModelManager:
    """Robust ModelManager for centralized model file management"""
    
    def __init__(self, base_path: str = None):
        self.base_path = base_path or "models"
        self.version_file = os.path.join(self.base_path, "versions.json")
        self.backup_dir = os.path.join(self.base_path, "backups")
        self.metrics_file = os.path.join(self.base_path, "metrics.json")
        
        # Create necessary directories
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Initialize or load version tracking
        self.versions = self._load_versions()
        self.metrics = self._load_metrics()
    
    def sanitize_symbol(self, symbol: str) -> str:
        """Sanitize symbol name for consistent file naming"""
        symbol = symbol.upper()
        symbol = symbol.replace('-USD', '')
        symbol = symbol.replace('USD', '')
        symbol = symbol.replace('-USDT', '')
        symbol = symbol.replace('USDT', '')
        return symbol
        
    def get_model_directory(self, symbol: str, timeframe: str) -> str:
        """Get standardized model directory path"""
        clean_symbol = self.sanitize_symbol(symbol)
        return os.path.join(self.base_path, f"{clean_symbol}_{timeframe}")
        
    def model_exists(self, symbol: str, timeframe: str) -> bool:
        """Check if model exists for given symbol and timeframe"""
        model_dir = self.get_model_directory(symbol, timeframe)
        model_path = os.path.join(model_dir, "model.pth")
        feature_scaler_path = os.path.join(model_dir, "feature_scaler.joblib")
        target_scaler_path = os.path.join(model_dir, "target_scaler.joblib")
        
        return (os.path.exists(model_path) and 
                os.path.exists(feature_scaler_path) and 
                os.path.exists(target_scaler_path))
    
    def delete_model(self, symbol: str, timeframe: str) -> bool:
        """Delete model files for given symbol and timeframe"""
        try:
            model_dir = self.get_model_directory(symbol, timeframe)
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
                logger.info(f"Deleted model directory: {model_dir}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting model for {symbol} {timeframe}: {e}")
            return False
        
    def _load_versions(self) -> Dict:
        """Load version information from file"""
        if os.path.exists(self.version_file):
            try:
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading version file: {str(e)}")
                return {}
        return {}
        
    def _save_versions(self):
        """Save version information to file"""
        try:
            with open(self.version_file, 'w') as f:
                json.dump(self.versions, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving version file: {str(e)}")
            
    def _load_metrics(self) -> Dict:
        """Load metrics history from file"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics file: {str(e)}")
                return {}
        return {}
        
    def _save_metrics(self):
        """Save metrics history to file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving metrics file: {str(e)}")
            
    def _update_metrics(self, symbol: str, timeframe: str, metrics: Dict[str, Any]):
        """Update metrics history for a model"""
        try:
            model_key = f"{symbol}_{timeframe}"
            if model_key not in self.metrics:
                self.metrics[model_key] = []
            
            self.metrics[model_key].append({
                'timestamp': datetime.now(pytz.UTC).isoformat(),
                'metrics': metrics
            })
            
            # Keep only last 10 entries
            if len(self.metrics[model_key]) > 10:
                self.metrics[model_key] = self.metrics[model_key][-10:]
                
            self._save_metrics()
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def get_model_history(self, symbol: str, timeframe: str) -> Dict:
        """Get model metrics history"""
        try:
            model_key = f"{symbol}_{timeframe}"
            return {
                'versions': self.versions.get(model_key, []),
                'metrics_history': self.metrics.get(model_key, [])
            }
        except Exception as e:
            logger.error(f"Error getting model history: {e}")
            return {'versions': [], 'metrics_history': []}
    
    def save_model_and_scalers(
        self, 
        model, 
        feature_scaler: Any, 
        target_scaler: Any, 
        symbol: str, 
        timeframe: str,
        config: Dict[str, Any],
        training_info: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save model, scalers, and metadata"""
        try:
            model_dir = self.get_model_directory(symbol, timeframe)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model with comprehensive checkpoint
            model_path = os.path.join(model_dir, "model.pth")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': config,
                'training_info': training_info or {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'training_date': datetime.now(pytz.UTC).isoformat(),
                },
                'symbol': symbol,
                'timeframe': timeframe,
                'version': '2.0',
                'metrics': metrics or {}
            }
            torch.save(checkpoint, model_path)
            
            # Save scalers
            feature_scaler_path = os.path.join(model_dir, "feature_scaler.joblib")
            target_scaler_path = os.path.join(model_dir, "target_scaler.joblib")
            
            joblib.dump(feature_scaler, feature_scaler_path)
            joblib.dump(target_scaler, target_scaler_path)
            
            # Update metrics tracking
            if metrics:
                self._update_metrics(symbol, timeframe, metrics)
            
            logger.info(f"Successfully saved model and scalers for {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model for {symbol} {timeframe}: {e}")
            return False
    
    def load_model_and_scalers(
        self, 
        symbol: str, 
        timeframe: str
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Dict], Optional[Dict]]:
        """
        Load model, scalers, and metadata
        Returns: (model, feature_scaler, target_scaler, config, training_info)
        """
        try:
            from ml_models.bilstm_predictor import BiLSTMWithAttention
            
            model_dir = self.get_model_directory(symbol, timeframe)
            model_path = os.path.join(model_dir, "model.pth")
            
            if not os.path.exists(model_path):
                logger.info(f"No model found for {symbol} {timeframe}")
                return None, None, None, None, None
                
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
                
            # Extract config
            config = checkpoint.get('config', {})
            if not config:
                logger.warning(f"No config found in checkpoint for {symbol} {timeframe}")
                return None, None, None, None, None
                
            # Initialize model
            model = BiLSTMWithAttention(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
                
            # Load scalers
            feature_scaler_path = os.path.join(model_dir, "feature_scaler.joblib")
            target_scaler_path = os.path.join(model_dir, "target_scaler.joblib")
            
            if not (os.path.exists(feature_scaler_path) and os.path.exists(target_scaler_path)):
                logger.warning(f"Scaler files missing for {symbol} {timeframe}")
                return None, None, None, None, None
                
            feature_scaler = joblib.load(feature_scaler_path)
            target_scaler = joblib.load(target_scaler_path)
                
            # Extract training info
            training_info = checkpoint.get('training_info', {})
            
            logger.info(f"Successfully loaded model and scalers for {symbol} {timeframe}")
            return model, feature_scaler, target_scaler, config, training_info
            
        except Exception as e:
            logger.error(f"Error loading model for {symbol} {timeframe}: {e}")
            return None, None, None, None, None
    
    def check_model_compatibility(self, symbol: str, timeframe: str, expected_config: Dict[str, Any]) -> bool:
        """Check if existing model is compatible with expected configuration"""
        try:
            _, _, _, config, _ = self.load_model_and_scalers(symbol, timeframe)
            
            if not config:
                return False
                
            # Check critical parameters
            critical_params = ['input_size', 'hidden_size', 'num_layers', 'dropout', 'lookback']
            for param in critical_params:
                if config.get(param) != expected_config.get(param):
                    logger.info(f"Model incompatible: {param} mismatch ({config.get(param)} vs {expected_config.get(param)})")
                    return False
                    
            # Check feature list compatibility
            existing_features = config.get('feature_list', [])
            expected_features = expected_config.get('feature_list', [])
            
            if set(existing_features) != set(expected_features):
                logger.info(f"Model incompatible: feature list mismatch")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking model compatibility: {e}")
            return False
