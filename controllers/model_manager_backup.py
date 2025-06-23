import os
import shutil
import json
import re
import joblib
from datetime import datetime
import pytz
import logging
from typing import Dict, Optional, Tuple, Any
import torch
import hashlib
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelManager:
    """Centralized, robust model file manager for all model operations"""
    
    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.version_file = self.base_path / "versions.json"
        self.backup_dir = self.base_path / "backups"
        self.metrics_file = self.base_path / "metrics.json"
        
        # Create necessary directories
        self.base_path.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize or load tracking
        self.versions = self._load_versions()
        self.metrics = self._load_metrics()
        
    def _sanitize_symbol(self, symbol: str) -> str:
        """Sanitize symbol for safe filesystem usage"""
        # Remove or replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', symbol.upper())
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        return sanitized
        
    def get_model_dir(self, symbol: str, timeframe: str) -> Path:
        """Get sanitized model directory path"""
        sanitized_symbol = self._sanitize_symbol(symbol)
        return self.base_path / f"{sanitized_symbol}_{timeframe}"
        
    def get_model_paths(self, symbol: str, timeframe: str) -> Dict[str, Path]:
        """Get all model-related file paths"""
        model_dir = self.get_model_dir(symbol, timeframe)
        return {
            'model_dir': model_dir,
            'model_file': model_dir / "model.pth",
            'feature_scaler': model_dir / "feature_scaler.joblib",
            'target_scaler': model_dir / "target_scaler.joblib",
            'config': model_dir / "config.json",
            'metrics': model_dir / "metrics.json"
        }
        
    def model_exists(self, symbol: str, timeframe: str) -> bool:
        """Check if a complete model exists (model + scalers)"""
        paths = self.get_model_paths(symbol, timeframe)
        required_files = ['model_file', 'feature_scaler', 'target_scaler']
        return all(paths[file].exists() for file in required_files)
        
    def delete_model(self, symbol: str, timeframe: str) -> bool:
        """Safely delete model and all associated files"""
        try:
            model_dir = self.get_model_dir(symbol, timeframe)
            if model_dir.exists():
                shutil.rmtree(model_dir)
                logger.info(f"Deleted model directory: {model_dir}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting model {symbol}_{timeframe}: {str(e)}")
            return False
            
    def save_model(self, model: Any, feature_scaler: Any, target_scaler: Any, 
                   config: Dict, metrics: Dict, symbol: str, timeframe: str) -> bool:
        """Save complete model with all components"""
        try:
            paths = self.get_model_paths(symbol, timeframe)
            paths['model_dir'].mkdir(parents=True, exist_ok=True)
            
            # Save model checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': config,
                'training_info': {
                    'training_time': datetime.now(pytz.UTC).isoformat(),
                    'best_metrics': metrics,
                    'symbol': symbol,
                    'timeframe': timeframe
                }
            }
            torch.save(checkpoint, paths['model_file'])
            
            # Save scalers
            joblib.dump(feature_scaler, paths['feature_scaler'])
            joblib.dump(target_scaler, paths['target_scaler'])
            
            # Save config and metrics separately for easy access
            with open(paths['config'], 'w') as f:
                json.dump(config, f, indent=2)
            with open(paths['metrics'], 'w') as f:
                json.dump(metrics, f, indent=2)
                
            logger.info(f"Successfully saved model {symbol}_{timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model {symbol}_{timeframe}: {str(e)}")
            return False
            
    def load_model(self, model_class: Any, symbol: str, timeframe: str) -> Optional[Tuple]:
        """Load complete model with all components"""
        try:
            if not self.model_exists(symbol, timeframe):
                return None
                
            paths = self.get_model_paths(symbol, timeframe)
            
            # Load checkpoint
            checkpoint = torch.load(paths['model_file'], map_location='cpu')
            config = checkpoint.get('config', {})
            training_info = checkpoint.get('training_info', {})
            
            # Initialize model
            model = model_class(**config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Load scalers
            feature_scaler = joblib.load(paths['feature_scaler'])
            target_scaler = joblib.load(paths['target_scaler'])
            
            # Get metrics
            metrics = training_info.get('best_metrics', {})
            
            logger.info(f"Successfully loaded model {symbol}_{timeframe}")
            return model, feature_scaler, target_scaler, config, metrics
            
        except Exception as e:
            logger.error(f"Error loading model {symbol}_{timeframe}: {str(e)}")
            return None
        
    def _load_versions(self) -> Dict:
        """Load version information from file"""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading version file: {str(e)}")
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
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics file: {str(e)}")
        return {}
        
    def _save_metrics(self):
        """Save metrics history to file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving metrics file: {str(e)}")
            
    def backup_model(self, symbol: str, timeframe: str) -> bool:
        """Create backup of existing model before replacement"""
        try:
            if not self.model_exists(symbol, timeframe):
                return False
                
            model_dir = self.get_model_dir(symbol, timeframe)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{self._sanitize_symbol(symbol)}_{timeframe}_{timestamp}"
            
            shutil.copytree(model_dir, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up model {symbol}_{timeframe}: {str(e)}")
            return False
            
    def list_models(self) -> Dict[str, Dict]:
        """List all available models with their metadata"""
        models = {}
        try:
            for model_dir in self.base_path.iterdir():
                if model_dir.is_dir() and '_' in model_dir.name:
                    parts = model_dir.name.split('_')
                    if len(parts) >= 2:
                        symbol = '_'.join(parts[:-1])
                        timeframe = parts[-1]
                        
                        paths = self.get_model_paths(symbol, timeframe)
                        if paths['metrics'].exists():
                            try:
                                with open(paths['metrics'], 'r') as f:
                                    metrics = json.load(f)
                                models[f"{symbol}_{timeframe}"] = {
                                    'symbol': symbol,
                                    'timeframe': timeframe,
                                    'metrics': metrics,
                                    'exists': self.model_exists(symbol, timeframe)
                                }
                            except Exception as e:
                                logger.error(f"Error reading metrics for {symbol}_{timeframe}: {str(e)}")
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
        return models

# Global model manager instance
model_manager = ModelManager()
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
            
            return version['hash']
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, symbol: str, timeframe: str, version_hash: Optional[str] = None) -> str:
        """Load specific model version"""
        try:
            model_key = f"{symbol}_{timeframe}"
            model_path = os.path.join(self.base_path, f"{model_key}.pth")
            
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
                        return backup_path
                        
                logger.warning(f"Version {version_hash} not found, using latest")
                
            return model_path
            
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