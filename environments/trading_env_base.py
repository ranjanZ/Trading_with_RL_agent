import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import yaml



logger = logging.getLogger(__name__)

@dataclass
class Trade:
    entry_time: int
    entry_price: float
    direction: int
    size: float
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None

class TradingEnvironment(gym.Env):
    """
    Trading Environment that follows the original working logic
    but adapted for gymnasium with historical data.
    
    Action values:
        -1 -> SELL
         0 -> HOLD
         1 -> BUY
    """

    def __init__(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        super().__init__()
    
        ############config loading ############
        # Load main config (trading parameters)
        with open("config/config.yaml", "r") as f:
            main_cfg = yaml.safe_load(f)

        # Load data config to get processed file path
        with open("config/data_config.yaml", "r") as f:
            data_cfg = yaml.safe_load(f)

        symbol = data_cfg['data']['symbols'][0]          # e.g., XAUUSD
        timeframe = "5m"                                  # choose one
        processed_file = Path(data_cfg['data']['processed_data_path']) / f"{symbol}_{timeframe}_features.csv"

        # Create environment config
        config = {
            "initial_balance": main_cfg['trading']['initial_balance'],
            "spread_pips": main_cfg['trading']['spread_pips'],
            "commission": main_cfg['trading']['commission'],
            "pip_value": main_cfg['trading'].get('pip_value', 0.01),
            "stop_loss_pips": 300,
            "take_profit_pips": 300,
            "max_hold_steps": 6,              # 30 minutes on 5m bars
            "trade_deadline_steps": 12,       # 60 minutes on 5m bars
            "data_path": str(processed_file),
            "feature_columns": main_cfg['data']['feature_columns'],   # from config.yaml
            "lookback_window": main_cfg['data']['lookback_window'],
        }
        ########################################


        # ----- Trading parameters (same as original) -----
        self.initial_balance = config.get("initial_balance", 10000.0)
        self.balance = self.initial_balance
        self.spread_pips = config.get("spread_pips", 0.2)
        self.commission = config.get("commission", 0.0)
        self.pip_value = config.get("pip_value", 0.01)
        self.stop_loss_pips = config.get("stop_loss_pips", 300)
        self.take_profit_pips = config.get("take_profit_pips", 300)
        self.max_hold_steps = config.get("max_hold_steps", 6)
        self.trade_deadline_steps = config.get("trade_deadline_steps", 12)
        self.active_trade_timeout_penalty = 0.2
        self.flat_wait_penalty = 0.05
        self.entry_step = None
        self.last_trade_step = None
        self.bars_since_last_trade = 0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0

        # ----- State tracking (same as original) -----
        self.current_position = 0          # -1: short, 0: flat, 1: long
        self.entry_price = 0.0
        self.position_size = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trade_history = []            # list of dicts: {type, entry, exit, pnl}

        # ----- Load historical data -----
        data_path = config.get("data_path")
        self.data = config.get("data")     # optionally pass DataFrame directly

        if self.data is None and data_path is not None:
            data_path = Path(data_path)
            if data_path.suffix == '.parquet':
                self.data = pd.read_parquet(data_path)
            else:
                self.data = pd.read_csv(data_path)
                if 'time' in self.data.columns:
                    self.data['time'] = pd.to_datetime(self.data['time'])
            # ADD THIS: Filter by date range
            if start_date and end_date and 'time' in self.data.columns:
                self.data = self.data[
                    (self.data['time'] >= pd.to_datetime(start_date)) & 
                    (self.data['time'] <= pd.to_datetime(end_date))
                ].reset_index(drop=True)
                print(f"Filtered to {len(self.data)} rows from {start_date} to {end_date}")
            
            logger.info(f"Loaded data from {data_path}: {len(self.data)} rows")

        elif self.data is not None:
            logger.info(f"Using provided data: {len(self.data)} rows")
        else:
            raise ValueError("Provide either 'data' or 'data_path' in config")

        # ----- Ensure required OHLCV columns exist -----
        required_cols = ['open', 'high', 'low', 'close']
        missing = [c for c in required_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")


        # ----- Feature columns for observation -----
        self.feature_columns = config.get("feature_columns")
        if self.feature_columns is None:
            # Try to use common technical indicators if they exist
            default_features = []
            self.feature_columns = [c for c in default_features if c in self.data.columns]
            if not self.feature_columns:
                self.feature_columns = ['close']
            logger.info(f"Auto-selected feature columns: {self.feature_columns}")

        self.lookback_window = config.get("lookback_window", 20)

        # ----- Observation & action spaces -----
        # +6 extra features: 3 for position one-hot, 3 for PnL info
        self.observation_dim = len(self.feature_columns) * (self.lookback_window + 1) + 6
        # Action space: -1 = SELL, 0 = HOLD, 1 = BUY
        self.action_space = spaces.Box(low=-1, high=1, shape=(), dtype=np.int32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )

        self.current_step = self.lookback_window
        self.max_steps = len(self.data) - 1
        self.last_trade_step = self.current_step
        self.bars_since_last_trade = 0


    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.current_position = 0
        self.entry_price = 0.0
        self.position_size = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trade_history.clear()
        self.entry_step = None
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.current_step = self.lookback_window
        self.last_trade_step = self.current_step
        self.bars_since_last_trade = 0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info




    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # action is directly -1, 0, or 1
        current_price = self._get_current_price()

        info = {
            'action': action,
            'action_name': self._get_action_names()[action],
            'prev_position': self.current_position,
            'trade_executed': False
        }
        reward = 0.0

        # ---- Action validation: enforce state machine ----
        valid_mask = self._get_valid_actions_mask()   # dict: { -1: bool, 0: bool, 1: bool }
        if not valid_mask.get(action, False):
            info['new_position'] = self.current_position
            self._update_unrealized_pnl(current_price)
            reward = -0.1  # penalty for invalid action
            obs = self._get_observation()
            info.update(self._get_info())
            self.current_step += 1
            return obs, reward, False, False, info

        # ---- Execute action with state machine logic ----
        if action == 1:  # BUY
            if self.current_position == 0:
                self._open_position(1, current_price)
                info['trade_executed'] = True
            elif self.current_position == -1:
                pnl = self._close_position(current_price)
                reward = pnl
                info['realized_pnl'] = pnl
                info['trade_executed'] = True
        
        elif action == -1:  # SELL
            if self.current_position == 0:
                self._open_position(-1, current_price)
                info['trade_executed'] = True
            elif self.current_position == 1:
                pnl = self._close_position(current_price)
                reward = pnl
                info['realized_pnl'] = pnl
                info['trade_executed'] = True
        
        elif action == 0:  # HOLD
            pass

        # Reward shaping for flat periods without trading
        if self.current_position == 0 and self.bars_since_last_trade >= self.trade_deadline_steps:
            reward -= self.flat_wait_penalty

        # Enforce stop loss / take profit / max hold time
        risk_reward_reward = self._enforce_trade_rules(current_price)
        reward += risk_reward_reward
        if risk_reward_reward != 0 and self.current_position == 0:
            info['trade_executed'] = True
            info['timeout_exit'] = True

        self._update_unrealized_pnl(current_price)
        self.current_step += 1

        if self.current_position == 0 and not info['trade_executed']:
            self.bars_since_last_trade += 1

        info['new_position'] = self.current_position
        info['unrealized_pnl'] = self.unrealized_pnl
        info['balance'] = self.balance
        info['total_pnl'] = self.unrealized_pnl + self.realized_pnl

        terminated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info.update(self._get_info())
        return obs, reward, terminated, False, info

    # ----- Helper methods (modified for state machine) -----
    def _get_valid_actions_mask(self):
        """
        Returns a dict mapping each possible action (-1, 0, 1) to a boolean
        indicating if it is allowed in the current state.
        Enforces order state machine:
        - Position 0 (flat): BUY, SELL, HOLD all allowed
        - Position 1 (long): cannot BUY, can SELL (close), can HOLD
        - Position -1 (short): can BUY (close), cannot SELL, can HOLD
        """
        if self.current_position == 0:
            return { -1: True, 0: True, 1: True }   # all allowed
        elif self.current_position == 1:
            return { -1: True, 0: True, 1: False }  # SELL allowed, BUY forbidden
        else:  # position == -1
            return { -1: False, 0: True, 1: True }  # BUY allowed, SELL forbidden

    def _get_action_names(self):
        """Maps action values to human-readable names."""
        return { -1: 'SELL', 0: 'HOLD', 1: 'BUY' }

    def _open_position(self, direction, price):
        spread_add = self.spread_pips * self.pip_value
        if direction == 1:
            self.entry_price = price + spread_add
            self.stop_loss_price = self.entry_price - self.stop_loss_pips * self.pip_value
            self.take_profit_price = self.entry_price + self.take_profit_pips * self.pip_value
        else:
            self.entry_price = price - spread_add
            self.stop_loss_price = self.entry_price + self.stop_loss_pips * self.pip_value
            self.take_profit_price = self.entry_price - self.take_profit_pips * self.pip_value
        self.current_position = direction
        self.position_size = 1.0
        self.entry_step = self.current_step
        self.last_trade_step = self.current_step
        self.bars_since_last_trade = 0

    def _close_position(self, price):
        if self.current_position == 0:
            return 0.0
        spread_sub = self.spread_pips * self.pip_value
        if self.current_position == 1:          # long
            exit_price = price - spread_sub
            pnl = (exit_price - self.entry_price) * self.position_size
        else:                                   # short
            exit_price = price + spread_sub
            pnl = (self.entry_price - exit_price) * self.position_size
        pnl -= self.commission
        self.realized_pnl += pnl
        self.balance += pnl
        self.trade_history.append({
            'type': 'LONG' if self.current_position == 1 else 'SHORT',
            'entry': self.entry_price,
            'exit': exit_price,
            'pnl': pnl,
            'duration_steps': self.current_step - self.entry_step if self.entry_step is not None else 0,
        })
        self.current_position = 0
        self.entry_price = 0.0
        self.position_size = 0.0
        self.entry_step = None
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        return pnl

    def _update_unrealized_pnl(self, price):
        if self.current_position == 0:
            self.unrealized_pnl = 0.0
        elif self.current_position == 1:
            self.unrealized_pnl = (price - self.entry_price) * self.position_size
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.position_size

    def _enforce_trade_rules(self, price) -> float:
        """Apply stop-loss/take-profit/hold-time enforcement and return extra reward/penalty."""
        if self.current_position == 0:
            return 0.0

        reward_adjustment = 0.0
        if self.current_position == 1:
            if price <= self.stop_loss_price:
                pnl = self._close_position(price)
                reward_adjustment += pnl - self.active_trade_timeout_penalty
            elif price >= self.take_profit_price:
                pnl = self._close_position(price)
                reward_adjustment += pnl + self.active_trade_timeout_penalty
        else:
            if price >= self.stop_loss_price:
                pnl = self._close_position(price)
                reward_adjustment += pnl - self.active_trade_timeout_penalty
            elif price <= self.take_profit_price:
                pnl = self._close_position(price)
                reward_adjustment += pnl + self.active_trade_timeout_penalty

        if self.current_position != 0 and self.entry_step is not None:
            hold_steps = self.current_step - self.entry_step
            if hold_steps >= self.max_hold_steps:
                pnl = self._close_position(price)
                reward_adjustment += pnl - self.active_trade_timeout_penalty

        return reward_adjustment

    def _get_observation(self) -> np.ndarray:
        if self.data is None or len(self.data) == 0:
            return np.zeros(self.observation_dim, dtype=np.float32)

        start = max(0, self.current_step - self.lookback_window)
        end = self.current_step + 1
        window = self.data.iloc[start:end]

        # Extract features as 2D array (time_steps x features)
        feature_matrix = []
        for col in self.feature_columns:
            if col in window.columns:
                feature_matrix.append(window[col].fillna(0).values)
            else:
                feature_matrix.append(np.zeros(len(window)))
        
        # Convert to numpy array (features x time)
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        
        # ----- NORMALIZE EACH FEATURE COLUMN INDEPENDENTLY -----
        # Normalize across time dimension for each feature
        for i in range(feature_matrix.shape[0]):
            mean = np.mean(feature_matrix[i])
            std = np.std(feature_matrix[i])
            if std > 1e-8:
                feature_matrix[i] = (feature_matrix[i] - mean) / std
            else:
                feature_matrix[i] = feature_matrix[i] - mean
        
        # Clip extreme values
        feature_matrix = np.clip(feature_matrix, -5.0, 5.0)
        
        # Flatten back to 1D
        features = feature_matrix.flatten()
        # ---------------------------------
        
        # Position one-hot
        pos = self.current_position
        pos_onehot = [1 if pos == 1 else 0, 1 if pos == -1 else 0, 1 if pos == 0 else 0]
        features = np.concatenate([features, pos_onehot])
        
        # PnL info (also normalize)
        pnl_features = np.array([self.unrealized_pnl, self.realized_pnl, 
                                self.unrealized_pnl + self.realized_pnl], dtype=np.float32)
        
        # Normalize PnL features
        pnl_mean = np.mean(pnl_features)
        pnl_std = np.std(pnl_features)
        if pnl_std > 1e-8:
            pnl_features = (pnl_features - pnl_mean) / pnl_std
        else:
            pnl_features = pnl_features - pnl_mean
        pnl_features = np.clip(pnl_features, -10.0, 10.0)
        
        features = np.concatenate([features, pnl_features])
        
        # Ensure fixed size
        if len(features) < self.observation_dim:
            features = np.pad(features, (0, self.observation_dim - len(features)))
        elif len(features) > self.observation_dim:
            features = features[:self.observation_dim]
        
        return features.astype(np.float32)


    def _get_observation_old(self) -> np.ndarray:
        
        if self.data is None or len(self.data) == 0:
            return np.zeros(self.observation_dim, dtype=np.float32)

        start = max(0, self.current_step - self.lookback_window)
        end = self.current_step + 1
        window = self.data.iloc[start:end]
        
        features = []
        for col in self.feature_columns:
            if col in window.columns:
                features.extend(window[col].fillna(0).values)
            else:
                features.extend([0] * len(window))
    
        
        # position one‑hot
        pos = self.current_position
        pos_onehot = [1 if pos == 1 else 0, 1 if pos == -1 else 0, 1 if pos == 0 else 0]
        features.extend(pos_onehot)

        # PnL info
        features.extend([self.unrealized_pnl, self.realized_pnl,
                         self.unrealized_pnl + self.realized_pnl])

        if len(features) < self.observation_dim:
            features.extend([0] * (self.observation_dim - len(features)))
        return np.array(features[:self.observation_dim], dtype=np.float32)

    def _get_current_price(self) -> float:
        if self.data is not None and self.current_step < len(self.data):
            return self.data.iloc[self.current_step]['close']
        return 0.0

    def _get_info(self) -> Dict:
        return {
            'position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.unrealized_pnl + self.realized_pnl,
            'balance': self.balance,
            'step': self.current_step,
            'trade_history': self.trade_history,
        }

    def get_state_info(self, price):
        """Compatibility with original state encoder"""
        self._update_unrealized_pnl(price)
        return self._get_info()
    


if __name__ == "__main__":

    print("Testing TradingEnvironment with random actions and state machine...")
    env = TradingEnvironment(start_date="2024-01-01",end_date="2024-12-31")
    

    # Run for more steps to ensure actual trades occur
    obs, info = env.reset()
    total_steps = 0

    episode_data = {
        'steps': [],
        'time': [],          # <-- NEW: store timestamps
        'prices': [],
        'open': [],
        'high': [],
        'low': [],
        'positions': [],
        'actions': [],
        'balance': [],
        'unrealized_pnl': [],
        'realized_pnl': [],
        'total_pnl': []
    }

    for step_num in range(500):   # run many steps to generate trades
        valid_mask = env._get_valid_actions_mask()
        valid_actions = [a for a, valid in valid_mask.items() if valid]
        action = np.random.choice(valid_actions)

        obs, reward, terminated, truncated, info = env.step(action)

        # Collect current bar data
        row = env.data.iloc[env.current_step]
        current_price = row['close']
        episode_data['steps'].append(step_num)
        episode_data['time'].append(row['time'])          # <-- store timestamp
        episode_data['prices'].append(current_price)
        episode_data['open'].append(row['open'])
        episode_data['high'].append(row['high'])
        episode_data['low'].append(row['low'])
        episode_data['positions'].append(info['position'])
        episode_data['actions'].append(action)
        episode_data['balance'].append(info['balance'])
        episode_data['unrealized_pnl'].append(info['unrealized_pnl'])
        episode_data['realized_pnl'].append(info['realized_pnl'])
        episode_data['total_pnl'].append(info['total_pnl'])

        if terminated:
            print(f"Episode terminated at step {step_num}")
            break

    print(f"\nFinal Balance: {info['balance']:.2f}")
    print(f"Total PnL: {info['total_pnl']:.2f}")
    print(f"Trades executed: {len(info['trade_history'])}")

    # --- Visualization with Candlestick Chart ---
    try:
        import mplfinance as mpf
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # Build OHLC DataFrame with DatetimeIndex (required by mplfinance)
        ohlc_df = pd.DataFrame({
            'Open':  episode_data['open'],
            'High':  episode_data['high'],
            'Low':   episode_data['low'],
            'Close': episode_data['prices']
        }, index=pd.DatetimeIndex(episode_data['time']))   # <-- fixes DatetimeIndex error

        # Detect entry/exit points from position changes
        pos_series = episode_data['positions']
        buy_entry_steps = []
        sell_entry_steps = []
        buy_exit_steps = []
        sell_exit_steps = []

        for i in range(1, len(pos_series)):
            if pos_series[i-1] == 0 and pos_series[i] == 1:
                buy_entry_steps.append(i)
            elif pos_series[i-1] == 0 and pos_series[i] == -1:
                sell_entry_steps.append(i)
            elif pos_series[i-1] == 1 and pos_series[i] == 0:
                buy_exit_steps.append(i)
            elif pos_series[i-1] == -1 and pos_series[i] == 0:
                sell_exit_steps.append(i)

        # Create marker arrays of same length as DataFrame, filled with NaN
        num_points = len(ohlc_df)
        markers = {
            'buy_entry': np.full(num_points, np.nan),
            'sell_entry': np.full(num_points, np.nan),
            'buy_exit': np.full(num_points, np.nan),
            'sell_exit': np.full(num_points, np.nan)
        }
        for step in buy_entry_steps:
            markers['buy_entry'][step] = episode_data['prices'][step]
        for step in sell_entry_steps:
            markers['sell_entry'][step] = episode_data['prices'][step]
        for step in buy_exit_steps:
            markers['buy_exit'][step] = episode_data['prices'][step]
        for step in sell_exit_steps:
            markers['sell_exit'][step] = episode_data['prices'][step]

        # Build additional plots for markers
        ap = []
        if buy_entry_steps:
            ap.append(mpf.make_addplot(markers['buy_entry'], type='scatter', marker='^', color='lime', markersize=100, panel=0))
        if sell_entry_steps:
            ap.append(mpf.make_addplot(markers['sell_entry'], type='scatter', marker='v', color='red', markersize=100, panel=0))
        if buy_exit_steps:
            ap.append(mpf.make_addplot(markers['buy_exit'], type='scatter', marker='o', color='blue', markersize=80, panel=0))
        if sell_exit_steps:
            ap.append(mpf.make_addplot(markers['sell_exit'], type='scatter', marker='o', color='orange', markersize=80, panel=0))

        # Plot candlestick + markers
        fig, axes = mpf.plot(ohlc_df,
                             type='candle',
                             style='charles',
                             addplot=ap if ap else None,
                             volume=False,
                             returnfig=True,
                             title='Trading Environment - Candlestick with Trade Markers')
        plt.savefig('Data/visulizaation/candlestick_trades.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Separate balance / PnL chart
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(episode_data['steps'], episode_data['balance'], label='Balance', color='blue')
        ax1.axhline(y=env.initial_balance, color='gray', linestyle='--', label='Initial')
        ax1.set_ylabel('Balance ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(episode_data['steps'], episode_data['total_pnl'], label='Total PnL', color='green')
        ax2.plot(episode_data['steps'], episode_data['realized_pnl'], label='Realized PnL', color='orange')
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('PnL ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('Data/visulizaation/balance_pnl.png', dpi=150, bbox_inches='tight')
        plt.show()

    except ImportError:
        print("mplfinance not installed. Falling back to simple line plot with markers.")
        import matplotlib.pyplot as plt

        # Fallback line plot with correct marker detection
        pos = episode_data['positions']
        steps = episode_data['steps']
        prices = episode_data['prices']

        buy_entry_steps = [i for i in range(1, len(pos)) if pos[i-1]==0 and pos[i]==1]
        sell_entry_steps = [i for i in range(1, len(pos)) if pos[i-1]==0 and pos[i]==-1]

        plt.figure(figsize=(14,8))
        plt.plot(steps, prices, color='black', label='Price')
        if buy_entry_steps:
            plt.scatter([steps[i] for i in buy_entry_steps],
                        [prices[i] for i in buy_entry_steps],
                        marker='^', color='green', s=100, label='BUY Entry')
        if sell_entry_steps:
            plt.scatter([steps[i] for i in sell_entry_steps],
                        [prices[i] for i in sell_entry_steps],
                        marker='v', color='red', s=100, label='SELL Entry')
        plt.legend()
        plt.savefig('Data/visulizaation/price_trades_fallback.png', dpi=150, bbox_inches='tight')
        plt.show()