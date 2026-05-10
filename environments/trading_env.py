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
    """

    def __init__(self,):
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
        self.observation_dim = len(self.feature_columns) * (self.lookback_window + 1) + 5
        self.action_space = spaces.Discrete(3)          # 0=BUY, 1=SELL, 2=HOLD
        # State machine:
        # Position 0 (flat): can BUY, SELL, or HOLD
        # Position 1 (long): can SELL (to close), HOLD
        # Position -1 (short): can BUY (to close), HOLD
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )

        self.current_step = self.lookback_window
        self.max_steps = len(self.data) - 1

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.current_position = 0
        self.entry_price = 0.0
        self.position_size = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trade_history.clear()

        self.current_step = self.lookback_window
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        current_price = self._get_current_price()

        info = {
            'action': action,
            'action_name': self._get_action_names()[action],
            'prev_position': self.current_position,
            'trade_executed': False
        }
        reward = 0.0

        # ---- Action validation: enforce state machine ----
        if not self._get_valid_actions_mask()[action]:
            info['new_position'] = self.current_position
            self._update_unrealized_pnl(current_price)
            reward = -0.1  # penalty for invalid action
            obs = self._get_observation()
            info.update(self._get_info())
            return obs, reward, False, False, info

        # ---- Execute action with state machine logic ----
        # BUY: open long if flat, or close short if in short position
        if action == 0:  # BUY
            if self.current_position == 0:
                # Open long position
                self._open_position(1, current_price)
                info['trade_executed'] = True
            elif self.current_position == -1:
                # Close short position
                pnl = self._close_position(current_price)
                reward = pnl
                info['realized_pnl'] = pnl
                info['trade_executed'] = True
        
        # SELL: open short if flat, or close long if in long position
        elif action == 1:  # SELL
            if self.current_position == 0:
                # Open short position
                self._open_position(-1, current_price)
                info['trade_executed'] = True
            elif self.current_position == 1:
                # Close long position
                pnl = self._close_position(current_price)
                reward = pnl
                info['realized_pnl'] = pnl
                info['trade_executed'] = True
        
        # HOLD: do nothing
        elif action == 2:  # HOLD
            pass

        self._update_unrealized_pnl(current_price)
        self.current_step += 1

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
        Enforce order state machine:
        - Position 0 (flat): can BUY, SELL, or HOLD
        - Position 1 (long): cannot BUY again, can SELL (close), or HOLD
        - Position -1 (short): can BUY (close), cannot SELL again, or HOLD
        """
        if self.current_position == 0:
            return [True, True, True]  # BUY, SELL, HOLD allowed
        elif self.current_position == 1:
            return [False, True, True]  # cannot BUY, can SELL (close), can HOLD
        else:  # position == -1
            return [True, False, True]  # can BUY (close), cannot SELL, can HOLD

    def _get_action_names(self):
        return {0: 'BUY', 1: 'SELL', 2: 'HOLD'}

    def _open_position(self, direction, price):
        spread_add = self.spread_pips * self.pip_value
        if direction == 1:
            self.entry_price = price + spread_add
        else:
            self.entry_price = price - spread_add
        self.current_position = direction
        self.position_size = 1.0

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
        })
        self.current_position = 0
        self.entry_price = 0.0
        self.position_size = 0.0
        return pnl

    def _update_unrealized_pnl(self, price):
        if self.current_position == 0:
            self.unrealized_pnl = 0.0
        elif self.current_position == 1:
            self.unrealized_pnl = (price - self.entry_price) * self.position_size
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.position_size

    def _get_observation(self) -> np.ndarray:
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
    env = TradingEnvironment()
    obs, info = env.reset()

    # Collect episode data for visualization
    episode_data = {
        'steps': [],
        'prices': [],
        'positions': [],
        'actions': [],
        'balance': [],
        'unrealized_pnl': [],
        'realized_pnl': [],
        'total_pnl': [],
    }

    for step_num in range(500):
        # Sample random action, but using valid actions mask
        valid_mask = env._get_valid_actions_mask()
        valid_actions = [i for i, valid in enumerate(valid_mask) if valid]
        action = np.random.choice(valid_actions)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Collect data
        current_price = env._get_current_price()
        episode_data['steps'].append(step_num)
        episode_data['prices'].append(current_price)
        episode_data['positions'].append(info['position'])
        episode_data['actions'].append(action)
        episode_data['balance'].append(info['balance'])
        episode_data['unrealized_pnl'].append(info['unrealized_pnl'])
        episode_data['realized_pnl'].append(info['realized_pnl'])
        episode_data['total_pnl'].append(info['total_pnl'])
        
        if step_num % 50 == 0:
            print(f"Step {step_num}: Price={current_price:.4f}, Balance={info['balance']:.2f}, "
                  f"Position={info['position']}, Total PnL={info['total_pnl']:.2f}")
        
        if terminated:
            print(f"Episode terminated at step {step_num}")
            break

    print(f"\nSimulation completed at step {step_num}!")
    print(f"Final Balance: {info['balance']:.2f}")
    print(f"Total PnL: {info['total_pnl']:.2f}")
    print(f"Realized PnL: {info['realized_pnl']:.2f}")
    print(f"Unrealized PnL: {info['unrealized_pnl']:.2f}")
    print(f"Total Trades: {len(info['trade_history'])}")
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        # Plot 1: Price and positions
        ax1 = axes[0]
        ax1.plot(episode_data['steps'], episode_data['prices'], label='Price', color='black', linewidth=2)
        
        # Mark buy and sell points
        buy_steps = [episode_data['steps'][i] for i in range(len(episode_data['steps'])) 
                     if episode_data['positions'][i] == 1 and (i == 0 or episode_data['positions'][i-1] == 0)]
        short_steps = [episode_data['steps'][i] for i in range(len(episode_data['steps'])) 
                       if episode_data['positions'][i] == -1 and (i == 0 or episode_data['positions'][i-1] == 0)]
        
        buy_prices = [episode_data['prices'][episode_data['steps'].index(s)] for s in buy_steps if s in episode_data['steps']]
        short_prices = [episode_data['prices'][episode_data['steps'].index(s)] for s in short_steps if s in episode_data['steps']]
        
        ax1.scatter(buy_steps[:len(buy_prices)], buy_prices, marker='^', color='green', s=100, label='BUY Entry', zorder=5)
        ax1.scatter(short_steps[:len(short_prices)], short_prices, marker='v', color='red', s=100, label='SELL Entry', zorder=5)
        
        ax1.set_ylabel('Price')
        ax1.set_title('Trading Environment - Price and Positions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Balance
        ax2 = axes[1]
        ax2.plot(episode_data['steps'], episode_data['balance'], label='Balance', color='blue', linewidth=2)
        ax2.axhline(y=env.initial_balance, color='gray', linestyle='--', label='Initial Balance')
        ax2.set_ylabel('Balance ($)')
        ax2.set_title('Account Balance Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: PnL Components
        ax3 = axes[2]
        ax3.plot(episode_data['steps'], episode_data['realized_pnl'], label='Realized PnL', color='green', linewidth=2)
        ax3.plot(episode_data['steps'], episode_data['total_pnl'], label='Total PnL', color='blue', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('PnL ($)')
        ax3.set_title('Profit and Loss Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Position
        ax4 = axes[3]
        colors = ['red' if p == -1 else 'green' if p == 1 else 'gray' for p in episode_data['positions']]
        ax4.bar(episode_data['steps'], episode_data['positions'], color=colors, width=1, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Position')
        ax4.set_yticks([-1, 0, 1])
        ax4.set_yticklabels(['SHORT', 'FLAT', 'LONG'])
        ax4.set_title('Position State Over Time (Red=SHORT, Green=LONG, Gray=FLAT)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('Data/visulizaation/trading_env_test_visualization.png', dpi=100, bbox_inches='tight')
        print("\nVisualization saved to: Data/visulizaation/trading_env_test_visualization.png")
        plt.close()
        
    except ImportError:
        print("Matplotlib not available for visualization")
