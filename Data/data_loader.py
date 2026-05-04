import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import pickle
import json
from sklearn.preprocessing import StandardScaler
import dask.dataframe as dd
from concurrent.futures import ThreadPoolExecutor
import tempfile
logger = logging.getLogger(__name__)

class DataLoader:
    """Unified data loader for historical, tick, and real-time data"""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['data']
        
        # Initialize paths
        self.raw_path = Path(self.config['raw_data_path'])
        self.processed_path = Path(self.config['processed_data_path'])
        self.cache_path = Path(self.config['cache_path'])
        
        # Create directories
        for path in [self.raw_path, self.processed_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)
            
        self.scalers = {}
        self.feature_columns = None
        
    def load_historical_data(
        self, 
        symbol: str = "XAUUSD", 
        timeframe: str = "5m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load historical OHLCV data"""
        
        # Determine date range
        if start_date is None:
            start_date = self.config['date_ranges']['train_start']
        if end_date is None:
            end_date = self.config['date_ranges']['test_end']
        
        # Build file pattern
        file_pattern = f"{symbol}_{timeframe}_*.csv"
        data_dir = self.raw_path / "historical"
        
        # Find all matching files
        files = sorted(data_dir.glob(file_pattern))
        
        if not files:
            logger.warning(f"No files found for {symbol} {timeframe}")
            return self._generate_sample_data(symbol, timeframe, start_date, end_date)
        
        # Load and concatenate data
        dfs = []
        for file in files:
            # Extract date from filename
            date_str = file.stem.split('_')[-1]
            file_date = datetime.strptime(date_str, "%Y%m%d")
            
            # Check if within range
            if start_date and file_date < datetime.strptime(start_date, "%Y-%m-%d"):
                continue
            if end_date and file_date > datetime.strptime(end_date, "%Y-%m-%d"):
                continue
                
            df = pd.read_csv(file)
            dfs.append(df)
        
        if not dfs:
            raise ValueError(f"No data found for {symbol} {timeframe} in date range")
        
        df = pd.concat(dfs, ignore_index=True)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        
        # Filter by exact date range
        if start_date:
            df = df[df['time'] >= start_date]
        if end_date:
            df = df[df['time'] <= end_date]
        
        logger.info(f"Loaded {len(df)} candles for {symbol} {timeframe}")
        return df
    
    def load_tick_data(
        self, 
        symbol: str = "XAUUSD",
        date: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> pd.DataFrame:
        """Load tick-level data"""
        
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        # Tick data path: data/raw/tick/2025/01/XAUUSD_ticks_20250101.parquet
        year = date[:4]
        month = date[4:6]
        
        tick_path = self.raw_path / "tick" / year / month / f"{symbol}_ticks_{date}.parquet"
        
        if not tick_path.exists():
            logger.warning(f"Tick data not found at {tick_path}")
            return self._generate_sample_ticks(symbol, date)
        
        df = pd.read_parquet(tick_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by time range
        if start_time:
            start_ts = datetime.strptime(f"{date} {start_time}", "%Y%m%d %H:%M:%S")
            df = df[df['timestamp'] >= start_ts]
        if end_time:
            end_ts = datetime.strptime(f"{date} {end_time}", "%Y%m%d %H:%M:%S")
            df = df[df['timestamp'] <= end_ts]
        
        logger.info(f"Loaded {len(df)} ticks for {symbol} on {date}")
        return df
    
    def load_streaming_data(self, callback=None):
        """Load real-time streaming data"""
        from kafka import KafkaConsumer
        import json
        
        consumer = KafkaConsumer(
            'market_data',
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='latest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        for message in consumer:
            data = message.value
            if callback:
                callback(data)
            yield data
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        fit_scaler: bool = False,
        save_processed: bool = True
    ) -> pd.DataFrame:
        """Prepare features for training"""
        
        from data.feature_engineering import FeatureEngineer
        
        engineer = FeatureEngineer(self.config)
        df_with_features = engineer.calculate_all_indicators(df)
        
        # Select feature columns
        feature_cols = self._get_feature_columns()
        available_cols = [col for col in feature_cols if col in df_with_features.columns]
        
        X = df_with_features[available_cols].fillna(0).values
        
        # Scale features
        if fit_scaler:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['feature_scaler'] = scaler
        else:
            if 'feature_scaler' not in self.scalers:
                raise ValueError("Scaler not fitted. Call with fit_scaler=True first")
            X_scaled = self.scalers['feature_scaler'].transform(X)
        
        df_with_features[available_cols] = X_scaled
        
        if save_processed:
            self._save_processed_data(df_with_features)
        
        return df_with_features
    
    def create_training_sequences(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
        target_col: str = 'action'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training"""
        
        X, y = [], []
        feature_cols = self._get_feature_columns()
        available = [col for col in feature_cols if col in df.columns]
        
        for i in range(lookback, len(df)):
            window = df[available].iloc[i-lookback:i+1].values
            X.append(window.flatten())
            
            if target_col in df.columns:
                y.append(df[target_col].iloc[i])
        
        X = np.array(X)
        y = np.array(y) if y else None
        
        return X, y
    
    def split_train_val_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Dict:
        """Split data into train/val/test sets"""
        
        test_ratio = 1 - train_ratio - val_ratio
        
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits = {
            'X_train': X[:train_end],
            'y_train': y[:train_end],
            'X_val': X[train_end:val_end],
            'y_val': y[train_end:val_end],
            'X_test': X[val_end:],
            'y_test': y[val_end:]
        }
        
        # Save splits
        output_dir = self.processed_path / "splits"
        output_dir.mkdir(exist_ok=True)
        
        for name, data in splits.items():
            np.save(output_dir / f"{name}.npy", data)
        
        # Save metadata
        metadata = {
            'train_size': len(splits['X_train']),
            'val_size': len(splits['X_val']),
            'test_size': len(splits['X_test']),
            'feature_dim': splits['X_train'].shape[1],
            'num_classes': len(np.unique(y))
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Data split: Train={len(splits['X_train'])}, "
                   f"Val={len(splits['X_val'])}, Test={len(splits['X_test'])}")
        
        return splits
    
    def _get_feature_columns(self) -> List[str]:
        """Get list of feature columns"""
        if self.feature_columns is None:
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'ema_5', 'ema_10', 'ema_20', 'ema_50',
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'atr_14', 'adx_14', 'plus_di_14', 'minus_di_14'
            ]
        return self.feature_columns
    
    def _save_processed_data(self, df: pd.DataFrame):
        """Save processed data to parquet"""
        output_path = self.processed_path / "features" / "processed_data.parquet"
        output_path.parent.mkdir(exist_ok=True)
        df.to_parquet(output_path, compression='snappy')
        logger.info(f"Saved processed data to {output_path}")
    
    def _generate_sample_data(self, symbol: str, timeframe: str, 
                               start_date: str, end_date: str) -> pd.DataFrame:
        """Generate sample data for testing"""
        logger.warning(f"Generating sample data for {symbol} {timeframe}")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Determine frequency
        freq_map = {'1m': '1min', '5m': '5min', '1h': '1H', '1d': '1D'}
        freq = freq_map.get(timeframe, '5min')
        
        date_range = pd.date_range(start=start, end=end, freq=freq)
        
        # Generate random walk prices
        np.random.seed(42)
        returns = np.random.normal(0, 0.001, len(date_range))
        prices = 2000 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'time': date_range,
            'open': prices * (1 + np.random.uniform(-0.0005, 0.0005, len(prices))),
            'high': prices * (1 + np.random.uniform(0, 0.001, len(prices))),
            'low': prices * (1 - np.random.uniform(0, 0.001, len(prices))),
            'close': prices,
            'volume': np.random.randint(100, 10000, len(prices))
        })
        
        # Save sample data
        sample_path = self.raw_path / "historical" / f"{symbol}_{timeframe}_sample.csv"
        sample_path.parent.mkdir(exist_ok=True)
        df.to_csv(sample_path, index=False)
        
        return df
    
    def _generate_sample_ticks(self, symbol: str, date: str) -> pd.DataFrame:
        """Generate sample tick data for testing"""
        logger.warning(f"Generating sample tick data for {symbol} on {date}")
        
        start = datetime.strptime(f"{date} 00:00:00", "%Y%m%d %H:%M:%S")
        end = datetime.strptime(f"{date} 23:59:59", "%Y%m%d %H:%M:%S")
        
        # Generate ticks (1 per second)
        timestamps = pd.date_range(start=start, end=end, freq='1s')
        
        # Random walk
        np.random.seed(42)
        returns = np.random.normal(0, 0.00001, len(timestamps))
        prices = 2000 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': np.random.randint(1, 10, len(timestamps)),
            'side': np.random.choice(['BUY', 'SELL'], len(timestamps))
        })
        
        # Save sample data
        year = date[:4]
        month = date[4:6]
        tick_path = self.raw_path / "tick" / year / month
        tick_path.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tick_path / f"{symbol}_ticks_{date}.parquet")
        
        return df


class DataStreamBuffer:
    """Efficient buffer for streaming data"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = []
        self.position = 0
        
    def add(self, data: Dict):
        """Add data point to buffer"""
        self.buffer.append(data)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def get_window(self, lookback: int) -> List[Dict]:
        """Get recent window of data"""
        start = max(0, len(self.buffer) - lookback)
        return self.buffer[start:]
    
    def get_feature_matrix(self, feature_cols: List[str]) -> np.ndarray:
        """Convert buffer to feature matrix"""
        if not self.buffer:
            return np.array([])
        
        features = []
        for data in self.buffer:
            row = [data.get(col, 0) for col in feature_cols]
            features.append(row)
        
        return np.array(features)
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()





if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DATA LOADER")
    print("=" * 60)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n📁 Test directory: {tmpdir}")
        
        # Test 1: Create mock data files
        print("\n📊 Test 1: Creating mock data files...")
        
        # Generate mock data
        np.random.seed(42)
        dates = pd.date_range(start='2025-01-01', periods=1000, freq='5min')
        prices = 2000 + np.cumsum(np.random.normal(0, 1, 1000))
        
        mock_df = pd.DataFrame({
            'time': dates,
            'open': prices,
            'high': prices + np.random.uniform(0, 2, 1000),
            'low': prices - np.random.uniform(0, 2, 1000),
            'close': prices + np.random.normal(0, 0.5, 1000),
            'volume': np.random.randint(100, 10000, 1000)
        })
        
        # Save mock data
        data_path = Path(tmpdir) / "mt5_data"
        data_path.mkdir()
        mock_df.to_parquet(data_path / "XAUUSD_5m_test.parquet")
        print(f"✓ Created mock data with {len(mock_df)} rows")
        
        # Test 2: Load data from files
        print("\n📂 Test 2: Loading data from files...")
        loader = DataLoader(data_path=str(data_path), use_mt5=False)
        df = loader.load_historical_data("XAUUSD", "5m")
        
        if df is not None:
            print(f"✓ Loaded {len(df)} candles")
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
        else:
            print("⚠️ No data loaded (expected if no MT5 connection)")
        
        # Test 3: Test with specified date range
        print("\n📅 Test 3: Loading with date filters...")
        df_filtered = loader.load_historical_data(
            "XAUUSD", "5m",
            start_date="2025-01-10",
            end_date="2025-01-20"
        )
        
        if df_filtered is not None:
            print(f"✓ Loaded {len(df_filtered)} candles in date range")
        else:
            print("⚠️ No data loaded with filters")
        
        # Test 4: Test feature preparation (mock)
        print("\n🔧 Test 4: Testing feature preparation...")
        if df is not None and len(df) > 0:
            # Add mock actions
            df['action'] = np.random.choice([0, 1, 2, 3], len(df))
            
            # Prepare features (simplified)
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
            X = df[feature_cols].fillna(0).values
            print(f"✓ Feature matrix shape: {X.shape}")
            
            # Create sequences
            lookback = 20
            X_seq = []
            for i in range(lookback, len(X)):
                X_seq.append(X[i-lookback:i+1].flatten())
            
            X_seq = np.array(X_seq)
            print(f"✓ Sequence shape: {X_seq.shape}")
        
        # Test 5: Test train/val/test split
        print("\n✂️ Test 5: Testing data split...")
        if df is not None and len(df) > 0 and 'action' in df.columns:
            X = np.random.randn(len(df), 50)
            y = df['action'].values
            
            train_end = int(0.7 * len(X))
            val_end = int(0.85 * len(X))
            
            splits = {
                'X_train': X[:train_end],
                'y_train': y[:train_end],
                'X_val': X[train_end:val_end],
                'y_val': y[train_end:val_end],
                'X_test': X[val_end:],
                'y_test': y[val_end:]
            }
            
            print(f"✓ Train size: {len(splits['X_train'])}")
            print(f"✓ Validation size: {len(splits['X_val'])}")
            print(f"✓ Test size: {len(splits['X_test'])}")
    
    print("\n" + "=" * 60)
    print("✅ DataLoader tests completed!")
    print("=" * 60)
