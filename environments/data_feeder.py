import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import deque
import ray
from ray.rllib.env.env_context import EnvContext

@ray.remote
class DataFeeder:
    """Scalable data feeder for streaming market data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.lookback_window = config.get("lookback_window", 20)
        self.buffer_size = config.get("buffer_size", 10000)
        self.feature_columns = config.get("feature_columns", [])
        
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.current_idx = 0
        self.feature_scaler = None
        
    def add_data(self, data: pd.DataFrame):
        """Add new data to buffer"""
        processed = self._process_data(data)
        self.data_buffer.extend(processed)
        
    def get_observation(self, idx: int) -> np.ndarray:
        """Get observation at specific index"""
        if idx < 0 or idx >= len(self.data_buffer):
            return np.zeros(self._get_feature_dim())
            
        # Get window of data
        start = max(0, idx - self.lookback_window)
        window = list(self.data_buffer)[start:idx+1]
        
        # Flatten and return
        flat_features = []
        for row in window:
            flat_features.extend(row)
            
        # Pad if necessary
        expected_len = self.lookback_window * self._get_feature_dim()
        if len(flat_features) < expected_len:
            flat_features.extend([0] * (expected_len - len(flat_features)))
            
        return np.array(flat_features[:expected_len], dtype=np.float32)
    
    def get_current_price(self, idx: int) -> float:
        """Get current price"""
        if idx >= 0 and idx < len(self.data_buffer):
            row = self.data_buffer[idx]
            # Assuming close price is at index 0
            return row[0]
        return 0.0
    
    def _process_data(self, df: pd.DataFrame) -> List[List[float]]:
        """Process raw data into features"""
        processed = []
        for idx, row in df.iterrows():
            features = []
            for col in self.feature_columns:
                if col in df.columns:
                    val = row[col]
                    if pd.isna(val):
                        val = 0.0
                    features.append(float(val))
            processed.append(features)
        return processed
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension"""
        return len(self.feature_columns)
    
    def get_length(self) -> int:
        """Get current buffer length"""
        return len(self.data_buffer)