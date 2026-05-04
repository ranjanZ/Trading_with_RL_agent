#!/usr/bin/env python3
"""Collect historical data from MT5 using config files."""

import sys
from pathlib import Path
import logging
import yaml
from datetime import datetime
import pandas as pd
import os
from typing import Optional, Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Data.connectors.mt5_connector import MT5Connector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_config(config_path: str = "config/data_config.yaml") -> dict:
    """Load data configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def collect_all_data(days_back = 20):
    """Collect historical data for all symbols and timeframes from config."""
    
    # Load data configuration
    try:
        data_cfg = load_data_config()
        logger.info("Loaded data_config.yaml")
    except FileNotFoundError:
        logger.error("config/data_config.yaml not found. Please create it.")
        return False
    
    if(days_back==None):
        date_ranges = data_cfg.get('data', {}).get('date_ranges', {})
        start_str = date_ranges.get('start_date')
        end_str = date_ranges.get('end_date')
        start = datetime.strptime(start_str, '%Y-%m-%d')
        end = datetime.strptime(end_str, '%Y-%m-%d')
        end=datetime.now()
        days_back=(end - start).days
        logger.info(f"Calculated days_back from date_ranges: {days_back} days")


    symbols = data_cfg.get('data', {}).get('symbols', ['XAUUSD'])
    timeframes = list(data_cfg.get('data', {}).get('timeframes', {}).keys())
    # Remove 'tick' if present because it's handled separately
    if 'tick' in timeframes:
        timeframes.remove('tick')
    
    
    logger.info(f"Symbols to collect: {symbols}")
    logger.info(f"Timeframes to collect: {timeframes}")
    logger.info(f"Days to collect: {days_back}")
    
    # Connect to MT5 using the connector (it reads mt5_config.yaml)
    logger.info("Initializing MT5 connector...")
    connector = MT5Connector()  # expects config/mt5_config.yaml
    if not connector.connected:
        logger.error("Failed to connect to MT5. Check your mt5_config.yaml and MT5 terminal.")
        return False
    
    # Data saver
    saver = MT5DataSaver(base_path=data_cfg['data']['raw_data_path'])
    
    collected_files = {}
    
    try:
        for symbol in symbols:
            # Temporarily set symbol in connector (if needed)
            # The MT5Connector uses its own config.symbol, we need to change it per symbol
            # Since MT5Connector stores config, we can update it dynamically
            connector.config.symbol = symbol
            logger.info(f"\n{'='*50}\nCollecting data for {symbol}\n{'='*50}")
            
            # Historical OHLCV data for each timeframe
            for tf in timeframes:
                logger.info(f"Fetching {tf} data for {symbol} (last {days_back} days)...")
                df = connector.get_historical_rates(timeframe=tf, days=days_back)
                if df is not None and len(df) > 0:
                    metadata = {
                        'symbol': symbol,
                        'timeframe': tf,
                        'days_collected': 'latest',
                        'start_date': df['time'].min().strftime('%Y-%m-%d %H:%M:%S'),
                        'end_date': df['time'].max().strftime('%Y-%m-%d %H:%M:%S'),
                        'total_bars': len(df),
                        'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    filepath = saver.save_rates(symbol, tf, df, metadata)
                    collected_files[f"{symbol}_{tf}"] = filepath
                    logger.info(f"Saved {len(df)} bars to {filepath}")
                else:
                    logger.warning(f"No data for {symbol} {tf}")
            
            # # Tick data (optional, limit to 5 days to avoid huge files)
            # logger.info(f"Fetching tick data for {symbol} (last 5 days)...")
            # tick_df = connector.get_ticks_range(days=min(5, days_back))
            # if tick_df is not None and len(tick_df) > 0:
            #     filepath = saver.save_ticks(symbol, tick_df, days=5)
            #     collected_files[f"{symbol}_ticks"] = filepath
            #     logger.info(f"Saved {len(tick_df)} ticks to {filepath}")
            # else:
            #     logger.warning(f"No tick data for {symbol}")
        
        logger.info("\n✅ Data collection complete!")
        logger.info(f"Total files saved: {len(collected_files)}")
        return True
        
    except Exception as e:
        logger.exception(f"Error during data collection: {e}")
        return False
    finally:
        connector.disconnect()



class MT5DataSaver:
    """Save MT5 data to files (unchanged – works with DataFrames)."""

    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = "./data/raw/"
        
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        logger.info(f"MT5DataSaver using base path: {self.base_path}")

    def save_rates(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        metadata: Dict = None,
    ) -> str:
        
        if df is None or len(df) == 0:
            logger.warning(f"No data to save for {symbol} {timeframe}")
            return None

        if metadata and "days_collected" in metadata:
            filename = f"{symbol}_{timeframe}_{metadata['days_collected']}_days.parquet"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timeframe}_{timestamp}.parquet"

        filepath = os.path.join(self.base_path, filename)
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(df)} {timeframe} candles to {filepath}")

        if metadata:
            meta_file = filepath.replace(".parquet", "_metadata.json")
            import json

            with open(meta_file, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        return filepath

    def save_ticks(self, symbol: str, df: pd.DataFrame, days: int) -> str:
        if df is None or len(df) == 0:
            logger.warning(f"No tick data to save for {symbol}")
            return None

        filename = f"{symbol}_ticks_{days}days.parquet"
        filepath = os.path.join(self.base_path, filename)
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(df)} ticks to {filepath}")
        return filepath




if __name__ == "__main__":
    success = collect_all_data(days_back=None)
    if success:
        print("\n✅ Historical data collected successfully!")
        print("You can now train your models using: python main.py --mode train")
    else:
        print("\n❌ Failed to collect data. Check your config and MT5 connection.")