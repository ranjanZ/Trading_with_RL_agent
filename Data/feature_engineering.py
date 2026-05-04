import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import talib
from scipy import signal
from pathlib import Path
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class FeatureEngineer:
    """Advanced feature engineering for trading data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.window_sizes = config.get('window_sizes', [5, 10, 20, 50])
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        
        df = df.copy()
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Trend indicators
        df = self._add_trend_indicators(df)
        
        # Momentum indicators
        df = self._add_momentum_indicators(df)
        
        # Volatility indicators
        df = self._add_volatility_indicators(df)
        
        # Volume indicators
        df = self._add_volume_indicators(df)
        
        # Custom features
        df = self._add_custom_features(df)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        
        # Returns
        df['return_1'] = df['close'].pct_change()
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price position
        for window in self.window_sizes:
            df[f'high_low_ratio_{window}'] = df['high'].rolling(window).max() / df['low'].rolling(window).min()
            df[f'close_position_{window}'] = (df['close'] - df['low'].rolling(window).min()) / (df['high'].rolling(window).max() - df['low'].rolling(window).min())
        
        # Spread
        df['spread'] = df['high'] - df['low']
        df['spread_pct'] = df['spread'] / df['close']
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators"""
        
        # Moving averages
        for window in self.window_sizes:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # MACD
        macd, signal_line, hist = self._calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal_line
        df['macd_hist'] = hist
        
        # Parabolic SAR
        df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
        
        # Ichimoku Cloud
        df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
        df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        
        # RSI
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # CCI
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        
        # ATR
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Donchian Channels
        for window in self.window_sizes:
            df[f'donchian_high_{window}'] = df['high'].rolling(window).max()
            df[f'donchian_low_{window}'] = df['low'].rolling(window).min()
            df[f'donchian_mid_{window}'] = (df[f'donchian_high_{window}'] + df[f'donchian_low_{window}']) / 2
        
        # Historical Volatility
        df['hv_10'] = df['return_1'].rolling(10).std() * np.sqrt(252)
        df['hv_20'] = df['return_1'].rolling(20).std() * np.sqrt(252)
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        
        if 'volume' not in df.columns:
            return df
        
        # Volume moving averages
        for window in self.window_sizes:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Volume Price Trend
        df['vpt'] = (df['volume'] * df['return_1']).fillna(0).cumsum()
        
        # Chaikin Money Flow
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        money_flow_volume = money_flow_multiplier * df['volume']
        df['cmf_20'] = money_flow_volume.rolling(20).sum() / df['volume'].rolling(20).sum()
        
        return df
    



    def _add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom engineered features"""
        
        # Price action patterns
        df['doji'] = abs(df['open'] - df['close']) <= (df['high'] - df['low']) * 0.1
        
        # Fix: Use bitwise '&' instead of 'and' for Series conditions
        hammer_condition1 = (df['high'] - df['low']) > 3 * abs(df['open'] - df['close'])
        hammer_condition2 = (df['close'] - df['low']) / (0.001 + df['high'] - df['low']) > 0.6
        hammer_condition3 = (df['open'] - df['low']) / (0.001 + df['high'] - df['low']) > 0.6
        
        df['hammer'] = hammer_condition1 & hammer_condition2 & hammer_condition3
        
        # Support and Resistance levels
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['dist_to_resistance'] = (df['resistance'] - df['close']) / df['close'] * 100
        df['dist_to_support'] = (df['close'] - df['support']) / df['close'] * 100
        
        # Market regime detection with fallback
        if 'sma_50' in df.columns:
            df['trend_strength'] = abs(df['close'] - df['sma_50']) / (df['sma_50'] + 1e-8)
        else:
            df['trend_strength'] = 0.0
        
        if 'hv_20' in df.columns:
            # Use pd.cut with proper bins
            df['volatility_regime'] = pd.cut(
                df['hv_20'].fillna(0), 
                bins=[-float('inf'), 0.1, 0.2, 0.3, float('inf')], 
                labels=['low', 'medium', 'high', 'extreme']
            )
        else:
            df['volatility_regime'] = 'medium'
        
        return df    

    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    



#################saving the processed features for later use#################
def load_config(config_path: str = "config/data_config.yaml") -> dict:
    """Load data configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_and_save_features():
    """Compute features for all symbols and timeframes and save to processed path."""
    
    # Load configuration
    config = load_config()
    data_config = config.get('data', {})
    
    # Get paths
    raw_data_path = data_config.get('raw_data_path', 'Data/raw/historical/')
    processed_data_path = data_config.get('processed_data_path', 'Data/processed/')
    
    # Get symbols and timeframes (exclude tick)
    symbols = data_config.get('symbols', ['XAUUSD'])
    timeframes = [tf for tf in data_config.get('timeframes', {}).keys() if tf != 'tick']
    
    # Get feature engineering config
    feature_config = {
        'window_sizes': data_config.get('features', {}).get('window_sizes', [5, 10, 20, 50])
    }
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(feature_config)
    
    raw_path = Path(raw_data_path)
    processed_path = Path(processed_data_path)
    
    logger.info(f"Raw data path: {raw_path}")
    logger.info(f"Processed data path: {processed_path}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Timeframes: {timeframes}")
    
    # Process each symbol and timeframe
    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"\nProcessing {symbol} - {timeframe}")
            
            # Find raw data files
            pattern = f"{symbol}_{timeframe}_*.parquet"
            files = list(raw_path.glob(pattern))
            
            if not files:
                pattern = f"{symbol}_{timeframe}_*.csv"
                files = list(raw_path.glob(pattern))
            
            if not files:
                logger.warning(f"No files found for {symbol} {timeframe}")
                continue
            
            # Load and combine all files
            dfs = []
            for file in files:
                if file.suffix == '.parquet':
                    df = pd.read_parquet(file)
                else:
                    df = pd.read_csv(file)
                    df['time'] = pd.to_datetime(df['time'])
                dfs.append(df)
            
            df = pd.concat(dfs, ignore_index=True)
            df = df.sort_values('time')
            logger.info(f"Loaded {len(df)} rows")
            
            # Compute features
            df_features = feature_engineer.calculate_all_indicators(df)
            logger.info(f"Computed {len(df_features.columns) - len(df.columns)} features")
            
            # Save to processed path
            output_dir = processed_path #/ symbol / timeframe
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{symbol}_{timeframe}_features.csv"
            df_features.to_csv(output_file, index=False)
            logger.info(f"Saved to {output_file}")

    


if __name__ == "__main__":
    # print("=" * 60)
    # print("TESTING FEATURE ENGINEERING")
    # print("=" * 60)
    
    # # Generate random price data
    # np.random.seed(42)
    # n_samples = 500
    # dates = pd.date_range(start='2025-01-01', periods=n_samples, freq='5min')
    
    # # Random walk prices
    # returns = np.random.normal(0, 0.001, n_samples)
    # prices = 2000 * np.exp(np.cumsum(returns))
    
    # df = pd.DataFrame({
    #     'time': dates,
    #     'open': prices * (1 + np.random.uniform(-0.0005, 0.0005, n_samples)),
    #     'high': prices * (1 + np.random.uniform(0, 0.001, n_samples)),
    #     'low': prices * (1 - np.random.uniform(0, 0.001, n_samples)),
    #     'close': prices,
    #     'volume': np.random.randint(100, 10000, n_samples)
    # })
    
    # print(f"\n📊 Generated {len(df)} rows of random price data")
    # print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # # Test feature engineering
    # print("\n🔧 Calculating technical indicators...")
    # engineer = FeatureEngineer(config={'window_sizes': [5, 10, 20]})
    # df_features = engineer.calculate_all_indicators(df)
    
    # print(f"✓ Original columns: {len(df.columns)}")
    # print(f"✓ After feature engineering: {len(df_features.columns)}")
    # print(f"✓ New features added: {len(df_features.columns) - len(df.columns)}")
    
    # # Show sample of new features
    # new_cols = [c for c in df_features.columns if c not in df.columns][:10]
    # print(f"\n📈 Sample new features: {new_cols}")
    
    # # Check for NaN values
    # nan_counts = df_features.isna().sum()
    # cols_with_nan = nan_counts[nan_counts > 0]
    # if len(cols_with_nan) > 0:
    #     print(f"\n⚠️ Columns with NaN values: {len(cols_with_nan)}")
    #     for col in cols_with_nan.head(5).index:
    #         print(f"  - {col}: {nan_counts[col]} NaNs")
    # else:
    #     print(f"\n✓ No NaN values found in features")
    
    # # Test specific indicators
    # print("\n📊 Testing specific indicators...")
    
    # # RSI
    # if 'rsi_14' in df_features.columns:
    #     print(f"✓ RSI range: {df_features['rsi_14'].min():.2f} - {df_features['rsi_14'].max():.2f}")
    
    # # MACD
    # if 'macd' in df_features.columns:
    #     print(f"✓ MACD range: {df_features['macd'].min():.2f} - {df_features['macd'].max():.2f}")
    
    # # Bollinger Bands
    # if 'bb_width' in df_features.columns:
    #     print(f"✓ BB Width range: {df_features['bb_width'].min():.4f} - {df_features['bb_width'].max():.4f}")
    
    # # ATR
    # if 'atr_14' in df_features.columns:
    #     print(f"✓ ATR range: {df_features['atr_14'].min():.2f} - {df_features['atr_14'].max():.2f}")
    
    # # Test rolling windows
    # print("\n🔄 Testing rolling window features...")
    # for window in [5, 10, 20]:
    #     sma_col = f'sma_{window}'
    #     if sma_col in df_features.columns:
    #         print(f"✓ SMA {window}: {df_features[sma_col].iloc[-1]:.2f}")
    
    # print("\n" + "=" * 60)
    # print("✅ Feature Engineering tests completed!")
    # print("=" * 60)    



    print("\n" + "=" * 60)
    print("COMPUTING AND SAVING FEATURES")
    print("=" * 60)
    compute_and_save_features()
    print("\n" + "=" * 60)
    print("✅ Feature Engineering tests completed!")
    print("=" * 60)
