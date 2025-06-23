import os
import shutil
import json
from datetime import datetime
import pytz
import logging
from typing import Dict, Optional, Tuple, Any
import torch
import hashlib
import pickle
import re
import numpy as np

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def _sanitize_filename(name: str) -> str:
    """Sanitizes a string to be safe for a filename."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '', name)

class ModelManager:
    """Manages model versioning, backups, and performance tracking"""
    
    def __init__(self, symbol: str = None, timeframe: str = None, base_path: str = "models"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.base_path = base_path
        self.version_file = os.path.join(base_path, "versions.json")
        self.backup_dir = os.path.join(base_path, "backups")
        self.metrics_file = os.path.join(base_path, "metrics.json")
        
        # Create necessary directories
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Initialize or load version tracking
        self.versions = self._load_versions()
        self.metrics = self._load_metrics()
        
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
                json.dump(self.versions, f, indent=4, cls=NumpyEncoder)
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
                json.dump(self.metrics, f, indent=4, cls=NumpyEncoder)
        except Exception as e:
            logger.error(f"Error saving metrics file: {str(e)}")
            
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA-256 hash of model file"""
        try:
            with open(model_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating model hash: {str(e)}")
            return ""

    def delete_model_files(self, symbol: str, timeframe: str):
        """Deletes all files associated with a specific model."""
        sanitized_symbol = _sanitize_filename(symbol)
        model_key = f"{sanitized_symbol}_{timeframe}"
        model_path = os.path.join(self.base_path, f"{model_key}.pth")
        
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                logger.info(f"Deleted existing model file: {model_path}")
            except OSError as e:
                logger.error(f"Error deleting file {model_path}: {e}")
            
    def save_model(self, model: torch.nn.Module, symbol: str, timeframe: str, 
                  metrics: Dict, model_path: str, scaler: Any = None) -> str:
        """Save model with version control"""
        try:
            # Generate version info
            version = {
                'timestamp': datetime.now(pytz.UTC).isoformat(),
                'metrics': metrics,
                'hash': None
            }
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'version_info': version,
                'metrics': metrics
            }, model_path)
            
            # Calculate hash
            version['hash'] = self._calculate_model_hash(model_path)
            
            # Update version tracking
            model_key = f"{symbol}_{timeframe}"
            if model_key not in self.versions:
                self.versions[model_key] = []
            self.versions[model_key].append(version)
            
            # Keep only last 5 versions
            if len(self.versions[model_key]) > 5:
                self.versions[model_key] = self.versions[model_key][-5:]
                
            self._save_versions()
            
            # Create backup
            backup_path = os.path.join(
                self.backup_dir, 
                f"{symbol}_{timeframe}_{version['timestamp']}.pth"
            )
            shutil.copy2(model_path, backup_path)
            
            # Update metrics history
            if model_key not in self.metrics:
                self.metrics[model_key] = []
            self.metrics[model_key].append({
                'timestamp': version['timestamp'],
                'metrics': metrics
            })
            self._save_metrics()
            
            # Save scaler if provided
            if scaler is not None:
                scaler_path = model_path.replace(".pth", "_scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                    
            return version['hash']
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, symbol: str, timeframe: str, version_hash: Optional[str] = None) -> Tuple[str, Any]:
        """Load specific model version"""
        try:
            model_key = f"{symbol}_{timeframe}"
            model_path = os.path.join(self.base_path, f"{model_key}.pth")
            scaler = None
            
            if version_hash:
                # Look for specific version in backups
                versions = self.versions.get(model_key, [])
                version_info = next(
                    (v for v in versions if v['hash'] == version_hash),
                    None
                )
                
                if version_info:
                    backup_path = os.path.join(
                        self.backup_dir,
                        f"{symbol}_{timeframe}_{version_info['timestamp']}.pth"
                    )
                    if os.path.exists(backup_path):
                        model_path = backup_path
                        
                        # Load corresponding scaler
                        scaler_path = model_path.replace(".pth", "_scaler.pkl")
                        if os.path.exists(scaler_path):
                            with open(scaler_path, 'rb') as f:
                                scaler = pickle.load(f)
                        
                        return model_path, scaler
                        
                logger.warning(f"Version {version_hash} not found, using latest")
                
            return model_path, scaler
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def get_model_history(self, symbol: str, timeframe: str) -> Dict:
        """Get model version history and performance metrics"""
        try:
            model_key = f"{symbol}_{timeframe}"
            return {
                'versions': self.versions.get(model_key, []),
                'metrics_history': self.metrics.get(model_key, [])
            }
        except Exception as e:
            logger.error(f"Error getting model history: {str(e)}")
            return {'versions': [], 'metrics_history': []}
            
    def should_retrain(self, symbol: str, timeframe: str, current_metrics: Dict) -> bool:
        """Determine if model should be retrained based on performance degradation"""
        try:
            model_key = f"{symbol}_{timeframe}"
            history = self.metrics.get(model_key, [])
            
            if not history:
                return True
                
            # Get average metrics from last 3 versions
            recent_metrics = history[-3:]
            avg_metrics = {
                'mae': sum(m['metrics']['mae'] for m in recent_metrics) / len(recent_metrics),
                'rmse': sum(m['metrics']['rmse'] for m in recent_metrics) / len(recent_metrics),
                'r2': sum(m['metrics']['r2'] for m in recent_metrics) / len(recent_metrics)
            }
            
            # Check for significant degradation (20% worse than average)
            degraded = (
                current_metrics['mae'] > avg_metrics['mae'] * 1.2 or
                current_metrics['rmse'] > avg_metrics['rmse'] * 1.2 or
                current_metrics['r2'] < avg_metrics['r2'] * 0.8
            )
            
            if degraded:
                logger.warning(f"Model performance degraded for {symbol}_{timeframe}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking retrain condition: {str(e)}")
            return True  # Retrain on error to be safe 

    def load_model_and_scalers(self, symbol: str, timeframe: str) -> Tuple[Any, Any, Any, Dict, Dict]:
        """Load model with scalers and configuration"""
        try:
            sanitized_symbol = _sanitize_filename(symbol)
            model_key = f"{sanitized_symbol}_{timeframe}"
            model_path = os.path.join(self.base_path, f"{model_key}.pth")
            
            if not os.path.exists(model_path):
                logger.info(f"No existing model found for {symbol} {timeframe}")
                return None, None, None, None, None
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Extract model configuration and training info
            config = checkpoint.get('config', {})
            if not config:
                config = checkpoint.get('model_config', {}) # Handle alternate key

            training_info = checkpoint.get('training_info', {})
            metrics = checkpoint.get('metrics', {})
            
            # Load the model architecture
            from controllers.prediction import BiLSTMWithAttention
            model = BiLSTMWithAttention(
                input_size=config.get('input_size', 26),
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.3)
            )
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Load scalers
            feature_scaler = checkpoint.get('feature_scaler')
            target_scaler = checkpoint.get('target_scaler')
            
            if feature_scaler is None or target_scaler is None:
                logger.warning(f"Missing scalers in saved model for {symbol} {timeframe}")
                return None, None, None, None, None
            
            logger.info(f"Successfully loaded model for {symbol} {timeframe} with R² = {metrics.get('r2', 'N/A')}")
            return model, feature_scaler, target_scaler, config, training_info
            
        except Exception as e:
            logger.error(f"Error loading model and scalers for {symbol} {timeframe}: {str(e)}")
            return None, None, None, None, None

    def save_model_and_scalers(self, model: torch.nn.Module, feature_scaler, target_scaler, 
                              symbol: str, timeframe: str, config: Dict, training_info: Dict, 
                              metrics: Dict) -> bool:
        """Save model with scalers and metadata"""
        try:
            sanitized_symbol = _sanitize_filename(symbol)
            model_key = f"{sanitized_symbol}_{timeframe}"
            model_path = os.path.join(self.base_path, f"{model_key}.pth")
            
            # Prepare checkpoint with everything needed
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'feature_scaler': feature_scaler,
                'target_scaler': target_scaler,
                'config': config,
                'training_info': training_info,
                'metrics': metrics,
                'timestamp': datetime.now(pytz.UTC).isoformat()
            }
            
            # Save checkpoint
            torch.save(checkpoint, model_path)
            
            # Update version tracking using existing method
            version = {
                'timestamp': checkpoint['timestamp'],
                'metrics': metrics,
                'hash': self._calculate_model_hash(model_path)
            }
            
            if model_key not in self.versions:
                self.versions[model_key] = []
            self.versions[model_key].append(version)
            
            # Keep only last 5 versions
            if len(self.versions[model_key]) > 5:
                self.versions[model_key] = self.versions[model_key][-5:]
                
            self._save_versions()
            
            # Update metrics history
            if model_key not in self.metrics:
                self.metrics[model_key] = []
            self.metrics[model_key].append({
                'timestamp': version['timestamp'],
                'metrics': metrics
            })
            self._save_metrics()
            
            logger.info(f"Successfully saved model for {symbol} {timeframe} with metrics: {metrics}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model and scalers for {symbol} {timeframe}: {str(e)}")
            return False

    def check_model_compatibility(self, symbol: str, timeframe: str, expected_config: Dict) -> bool:
        """Check if existing model is compatible with expected configuration"""
        try:
            model_key = f"{symbol}_{timeframe}"
            model_path = os.path.join(self.base_path, f"{model_key}.pth")
            
            if not os.path.exists(model_path):
                return False
            
            # Load model to check configuration
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            config = checkpoint.get('config', {})
            
            # Check critical parameters
            critical_params = ['input_size', 'hidden_size', 'num_layers', 'lookback']
            for param in critical_params:
                if config.get(param) != expected_config.get(param):
                    logger.warning(f"Model incompatible: {param} mismatch ({config.get(param)} vs {expected_config.get(param)})")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking model compatibility for {symbol} {timeframe}: {str(e)}")
            return False

    def delete_model(self, symbol: str, timeframe: str) -> bool:
        """Delete existing model for symbol/timeframe"""
        try:
            model_key = f"{symbol}_{timeframe}"
            model_path = os.path.join(self.base_path, f"{model_key}.pth")
            
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"Deleted existing model for {symbol} {timeframe}")
                
                # Also remove from version tracking
                if model_key in self.versions:
                    del self.versions[model_key]
                    self._save_versions()
                    
                if model_key in self.metrics:
                    del self.metrics[model_key]
                    self._save_metrics()
                    
                return True
            else:
                logger.info(f"No existing model to delete for {symbol} {timeframe}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting model for {symbol} {timeframe}: {str(e)}")
            return False

    def get_latest_metrics(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get the latest training metrics for a model"""
        try:
            model_key = f"{symbol}_{timeframe}"
            model_path = os.path.join(self.base_path, f"{model_key}.pth")
            
            if not os.path.exists(model_path):
                return None
            
            # Load metrics from checkpoint
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            metrics = checkpoint.get('metrics', {})
            
            # Add trained_on_demand flag based on timestamp
            training_info = checkpoint.get('training_info', {})
            training_date = training_info.get('training_date')
            if training_date:
                # If model was trained recently (within last hour), mark as on-demand
                from datetime import datetime, timedelta
                training_time = datetime.fromisoformat(training_date.replace('Z', '+00:00'))
                now = datetime.now(pytz.UTC)
                metrics['trained_on_demand'] = (now - training_time).total_seconds() < 3600
            else:
                metrics['trained_on_demand'] = False
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting latest metrics for {symbol} {timeframe}: {str(e)}")
            return None