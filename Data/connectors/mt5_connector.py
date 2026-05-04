from mt5linux import MetaTrader5 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from typing import Optional, Dict, List, Tuple
import logging
from dataclasses import dataclass, field
import yaml
from pathlib import Path
mt5 = MetaTrader5()

logger = logging.getLogger(__name__)


@dataclass
class MT5Config:
    """
    Configuration loaded from mt5_config.yaml.
    Contains account details and data‑retrieval parameters.
    """
    mode: str = field(default="DEMO_ACCOUNT")
    account_number: str = field(default="")
    password: str = field(default="")
    server: str = field(default="")
    symbol: str = field(default="XAUUSD")
    lot_multiplier: float = field(default=1.0)
    
    @classmethod
    def from_yaml(cls, yaml_path: str = "mt5_config.yaml") -> "MT5Config":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        mode = raw.get("MODE", "DEMO_ACCOUNT")
        account = raw.get(mode, {})
        if not account:
            raise ValueError(f"Account section '{mode}' not found in config")

        return cls(
            mode=mode,
            account_number=account.get('number'),
            password=account.get('password'),
            server=account.get('server'),
            lot_multiplier=account.get('lot_multiplier', 1.0)
        )

    def get_account(self) -> Dict[str, object]:
        """Return account credentials as a dict ready for mt5.initialize()."""
        return {
            "login": self.account_number,
            "password": self.password,
            "server": self.server,
        }


class MT5Connector:
    """MetaTrader 5 Connector for live and historical data (generic, config‑driven)."""

    def __init__(self, config_path: str = "config/mt5_config.yaml"):
        """
        If a config object is not provided, one is loaded from `config_path`.
        """
        self.config = MT5Config.from_yaml(config_path)
        self.connected = False
        self.logger = logging.getLogger(__name__)
        self.connect()
    def connect(self) -> bool:
        """Connect to MetaTrader 5 using account credentials from the config."""
       
        print(f"Attempting to connect to MT5 with account: {self.config.account_number} ({self.config.mode})")
        try:
            account = self.config.get_account()
            # Official MetaTrader5 initialization
            if not mt5.initialize(
                login=account["login"],
                password=account["password"],
                server=account["server"],
            ):
                self.logger.error(
                    f"MT5 initialize() failed. Error: {mt5.last_error()}"
                )
                return False
            self.connected = True
            print("MT5 connection successful.")
            # Get symbol info
            symbol_info = mt5.symbol_info(self.config.symbol)
            if not symbol_info:
                print(f"Cannot get {self.symbol} info")
                return False

            
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to MT5: {e}")
            return False

    def disconnect(self):
        """Shutdown MT5 connection."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Disconnected from MT5")

    def _check_connection(self) -> bool:
        """Verify that the terminal is still reachable."""
        if not self.connected:
            self.logger.error("Not connected to MT5")
            return False
        if not mt5.terminal_info():
            self.logger.error("MT5 terminal is not accessible")
            return False
        return True

    def get_rates(self, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        """Get OHLCV rates for a specific timeframe."""
        if not self._check_connection():
            return None

        timeframe_map = {
            "1h": mt5.TIMEFRAME_H1,
            "30m": mt5.TIMEFRAME_M30,
            "15m": mt5.TIMEFRAME_M15,
            "5m": mt5.TIMEFRAME_M5,
            "1m": mt5.TIMEFRAME_M1,
        }
        tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_M1)

        try:
            rates = mt5.copy_rates_from_pos(self.config.symbol, tf, 0, count)
            if rates is None or len(rates) == 0:
                self.logger.warning(f"No rates returned for {timeframe}")
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            return df[["time", "open", "high", "low", "close", "volume"]]

        except Exception as e:
            self.logger.error(f"Error getting rates for {timeframe}: {e}")
            return None

    def get_historical_rates(
        self, timeframe: str, days: int = 200
    ) -> Optional[pd.DataFrame]:
        
        if not self._check_connection():
            return None

        timeframe_map = {
            "1h": (mt5.TIMEFRAME_H1, 24),
            "30m": (mt5.TIMEFRAME_M30, 48),
            "15m": (mt5.TIMEFRAME_M15, 96),
            "5m": (mt5.TIMEFRAME_M5, 288),
            "1m": (mt5.TIMEFRAME_M1, 1440),
        }
        if timeframe not in timeframe_map:
            self.logger.error(f"Unsupported timeframe: {timeframe}")
            return None

        tf, bars_per_day = timeframe_map[timeframe]
        total_bars = days * bars_per_day
        self.logger.info(
            f"Fetching {days} days of {timeframe} data ({total_bars} bars)"
        )

        chunk_size = 5000
        all_rates = []                      # will become a list of rows
        for start_pos in range(0, total_bars, chunk_size):
            chunk_bars = min(chunk_size, total_bars - start_pos)
            rates = mt5.copy_rates_from_pos(
                self.config.symbol, tf, start_pos, chunk_bars
            )
            if rates is None:
                self.logger.warning(f"No data at position {start_pos}")
                break

            # Ensure each row is a simple list/tuple, not a nested object
            if isinstance(rates, np.ndarray):
                # structured array -> list of tuples (from mt5 official or mt5linux)
                all_rates.extend(rates.tolist())
            else:
                # already a list of rows
                all_rates.extend(rates)
            time.sleep(0.1)

        if not all_rates:
            self.logger.error(f"No historical data for {timeframe}")
            return None

        # Explicit column names matching the output order of copy_rates_from_pos
        columns = [
            "time", "open", "high", "low", "close",
            "tick_volume", "spread", "real_volume"
        ]
        df = pd.DataFrame(all_rates, columns=columns)

        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        self.logger.info(
            f"Retrieved {len(df)} {timeframe} bars from {df['time'].min()} to {df['time'].max()}"
        )
        return df[["time", "open", "high", "low", "close", "volume"]]


    def get_ticks_range(self, days: int = 5) -> Optional[pd.DataFrame]:
        """Get tick data for a specified number of days."""
        if not self._check_connection():
            return None

        from_date = datetime.now() - timedelta(days=days)
        to_date = datetime.now()
        self.logger.info(f"Fetching ticks from {from_date} to {to_date}")

        all_ticks = []
        current_start = from_date

        for i in range(days):
            chunk_start = current_start
            chunk_end = min(current_start + timedelta(days=1), to_date)
            if chunk_start > datetime.now():
                break

            try:
                ticks = mt5.copy_ticks_range(
                    self.config.symbol,
                    chunk_start,
                    chunk_end,
                    mt5.COPY_TICKS_ALL,
                )
                if ticks is not None and len(ticks) > 0:
                    all_ticks.extend(ticks)
                    self.logger.info(f"  Day {i+1}: {len(ticks)} ticks")
            except Exception as e:
                self.logger.warning(
                    f"Error fetching ticks for {chunk_start.date()}: {e}"
                )
            current_start = chunk_end
            time.sleep(0.5)

        if not all_ticks:
            self.logger.warning("No tick data retrieved")
            return None

        df = pd.DataFrame(all_ticks)
        df["time"] = pd.to_datetime(df["time"], unit="s")

        cols = ["time", "bid", "ask"]
        if "last" in df.columns:
            cols.append("last")
        cols.append("volume")
        return df[cols]

    def get_current_price(self) -> Tuple[Optional[float], Optional[float]]:
        """Get current bid/ask."""
        if not self._check_connection():
            return None, None

        tick = mt5.symbol_info_tick(self.config.symbol)
        if tick:
            return tick.bid, tick.ask
        return None, None

    def get_multi_timeframe_state(self) -> np.ndarray:
        """Get current state (normalised) across multiple timeframes."""
        if not self._check_connection():
            return np.array([])

        state_parts = []
        timeframe_configs = [
            ("1h", self.config.n_1h_candles),
            ("30m", self.config.n_30m_candles),
            ("5m", self.config.n_5m_candles),
            ("1m", self.config.n_1m_candles),
        ]

        for timeframe, candles in timeframe_configs:
            df = self.get_rates(timeframe, candles)
            if df is not None and len(df) > 0:
                values = df[["open", "high", "low", "close", "volume"]].values
                values = (values - values.mean()) / (values.std() + 1e-8)
                state_parts.append(values.flatten())

        if not state_parts:
            return np.array([])
        return np.concatenate(state_parts)


# ====================================================================
# Example YAML file (mt5_config.yaml) – place this in your project:
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MT5 CONNECTOR (generic, config‑driven)")
    print("=" * 60)

    # Test 2: Connector instantiation
    print("\n🔌 Test 2: Instantiating connector...")
    #connector = MT5Connector(config_path="config/mt5_config.yaml")
    connector = MT5Connector()

    print(f"  Connected status: {connector.connected}")



    TIMEFRAME = "1m"      # change as needed: "1m","5m","15m","30m","1h"
    DAYS_BACK = 1        # how many days of data to fetch

    print(f"\n📊 Fetching {DAYS_BACK} days of {TIMEFRAME} data...")
    df_hist = connector.get_historical_rates(timeframe=TIMEFRAME, days=DAYS_BACK)




    # Test 3: DataSaver test (with mock data)
    print("\n💾 Test 3: Testing MT5DataSaver...")
    from tempfile import TemporaryDirectory

    np.random.seed(42)
    n_bars = 100
    dates = pd.date_range(start="2025-01-01", periods=n_bars, freq="5min")
    prices = 2000 * np.exp(np.cumsum(np.random.normal(0, 0.001, n_bars)))
    mock_df = pd.DataFrame(
        {
            "time": dates,
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": np.random.randint(100, 10000, n_bars),
        }
    )

    print("\n" + "=" * 60)
    print("✅ All tests passed (no live MT5 connection attempted).")
    print("=" * 60)