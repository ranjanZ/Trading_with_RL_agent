import numpy as np
from typing import Tuple, Dict, List
from environments.trading_env import TradingEnvironment

class CompetitiveTradingEnv:
    """
    Two‑agent wrapper around a single‑asset TradingEnvironment.
    Both agents trade the same asset simultaneously.

    - Uses the base environment only for market data and feature construction.
    - Maintains separate positions, entry prices, PnL, and trade histories.
    - Rewards can be 'independent' or 'zero_sum'.
    - Optionally appends opponent's position one‑hot to each agent's observation.
    """

    def __init__(self, base_env: TradingEnvironment,
                 reward_mode: str = 'zero_sum',        # 'independent' or 'zero_sum'
                 include_opponent_position: bool = True,
                 transaction_cost: float = 0.0):
        self.base_env = base_env
        self.reward_mode = reward_mode
        self.include_opponent_position = include_opponent_position
        self.transaction_cost = transaction_cost

        # Observation dimensions – the base environment appends
        # 3 pos one‑hot + 3 PnL info at the end of its observation.
        self.base_obs_dim = base_env.observation_space.shape[0]
        self.market_dim = self.base_obs_dim - 6        # features only
        self.pos_dim = 3                                # [long, short, flat]
        self.pnl_dim = 3                                # [unrealised, realised, total]
        self.opponent_dim = 3 if include_opponent_position else 0
        self.obs_dim = self.market_dim + self.pos_dim + self.pnl_dim + self.opponent_dim

        # Agent states
        self.positions = [0, 0]            # -1, 0, 1
        self.entry_prices = [0.0, 0.0]
        self.pnls = [0.0, 0.0]             # total (realised + unrealised)
        self.realized_pnls = [0.0, 0.0]
        self.trade_histories = [[], []]

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset both agents and the underlying data."""
        base_obs, _ = self.base_env.reset()
        self.positions = [0, 0]
        self.entry_prices = [0.0, 0.0]
        self.pnls = [0.0, 0.0]
        self.realized_pnls = [0.0, 0.0]
        self.trade_histories = [[], []]
        return self._agent_obs(0, base_obs), self._agent_obs(1, base_obs)

    def step(self, actions: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray,
                                                        Tuple[float, float], bool, Dict]:
        """
        Execute one step for both agents.
        Args:
            actions: (action_agent0, action_agent1) each in {-1, 0, 1}
        Returns:
            next_obs0, next_obs1, (reward0, reward1), done, info
        """
        action0, action1 = actions
        prev_price = self.base_env._get_current_price()

        # Move the base environment forward by one HOLD action – we only need the next price
        base_obs, _, terminated, truncated, _ = self.base_env.step(0)
        current_price = self.base_env._get_current_price()
        price_change = current_price - prev_price
        done = terminated or truncated

        # Process each agent’s action
        pnl_changes = []
        for agent_id, action in enumerate([action0, action1]):
            pnl = self._process(agent_id, action, price_change, current_price)
            pnl_changes.append(pnl)

        # Compute rewards
        if self.reward_mode == 'zero_sum':
            reward0 = pnl_changes[0] - pnl_changes[1]
            reward1 = pnl_changes[1] - pnl_changes[0]
        else:
            reward0 = pnl_changes[0]
            reward1 = pnl_changes[1]

        info = {
            'pnl_agent0': self.pnls[0],
            'pnl_agent1': self.pnls[1],
            'trade_history_agent0': self.trade_histories[0],
            'trade_history_agent1': self.trade_histories[1],
        }

        # Build agent‑specific observations from the new base observation
        obs0 = self._agent_obs(0, base_obs)
        obs1 = self._agent_obs(1, base_obs)
        return obs0, obs1, (reward0, reward1), done, info

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _process(self, agent_id, action, price_change, current_price):
        pos = self.positions[agent_id]
        pnl = 0.0

        if action == 1:          # BUY
            if pos == 0:
                self.positions[agent_id] = 1
                self.entry_prices[agent_id] = current_price
                pnl = -self.transaction_cost * current_price
            elif pos == -1:
                pnl = -price_change - self.transaction_cost * current_price
                self._close_trade(agent_id, pnl)
            else:                # already long → mark to market
                pnl = price_change

        elif action == -1:       # SELL
            if pos == 0:
                self.positions[agent_id] = -1
                self.entry_prices[agent_id] = current_price
                pnl = -self.transaction_cost * current_price
            elif pos == 1:
                pnl = price_change - self.transaction_cost * current_price
                self._close_trade(agent_id, pnl)
            else:                # already short → mark to market
                pnl = -price_change

        else:                    # HOLD
            if pos == 1:
                pnl = price_change
            elif pos == -1:
                pnl = -price_change
            # flat → pnl = 0

        self.pnls[agent_id] += pnl
        return pnl

    def _close_trade(self, agent_id, pnl):
        self.realized_pnls[agent_id] += pnl
        self.trade_histories[agent_id].append({'pnl': pnl})
        self.positions[agent_id] = 0

    def _agent_obs(self, agent_id, base_obs):
        """
        Build observation for one agent using the base observation
        (which already contains normalised market features).
        """
        # Market features: first self.market_dim elements
        market = base_obs[:self.market_dim].copy()

        # Agent's own position one‑hot
        pos = self.positions[agent_id]
        own_pos = np.array([1 if pos == 1 else 0,
                            1 if pos == -1 else 0,
                            1 if pos == 0 else 0], dtype=np.float32)

        # Unrealised PnL
        cur_price = self.base_env._get_current_price()
        if pos == 1:
            unreal = cur_price - self.entry_prices[agent_id]
        elif pos == -1:
            unreal = self.entry_prices[agent_id] - cur_price
        else:
            unreal = 0.0
        realised = self.realized_pnls[agent_id]
        total = unreal + realised

        # Simple scaling (can be improved)
        pnl_info = np.clip(np.array([unreal, realised, total], dtype=np.float32) / 100.0,
                           -10.0, 10.0)

        # Opponent position one‑hot (optional)
        if self.include_opponent_position:
            opp_pos = self.positions[1 - agent_id]
            opp_onehot = np.array([1 if opp_pos == 1 else 0,
                                   1 if opp_pos == -1 else 0,
                                   1 if opp_pos == 0 else 0], dtype=np.float32)
            return np.concatenate([market, own_pos, pnl_info, opp_onehot]).astype(np.float32)
        return np.concatenate([market, own_pos, pnl_info]).astype(np.float32)

    # -----------------------------------------------------------------
    # Visualisation helper (same style as single‑agent TradingEnvironment)
    # -----------------------------------------------------------------
    @staticmethod
    def visualize_episode(episode_data: Dict, title: str = "Competitive Trading Environment"):
        """
        Plot candlestick charts with trade markers:
          - Combined chart (both agents)
          - Separate chart for Agent 0 only
          - Separate chart for Agent 1 only
        Also plots PnL comparison.

        Markers:
            Buy Entry   : green up‑triangle
            Sell Entry  : red down‑triangle
            Buy Exit    : blue circle (close long)
            Sell Exit   : orange circle (close short)

        episode_data must contain keys:
            'time', 'open', 'high', 'low', 'prices', 'steps',
            'positions' -> dict {0: list, 1: list},
            'pnl' -> dict {0: list, 1: list}
        """
        try:
            import mplfinance as mpf
            import matplotlib.pyplot as plt
            import pandas as pd

            # Build OHLC DataFrame
            ohlc_df = pd.DataFrame({
                'Open':  episode_data['open'],
                'High':  episode_data['high'],
                'Low':   episode_data['low'],
                'Close': episode_data['prices']
            }, index=pd.DatetimeIndex(episode_data['time']))

            # Helper to find all four marker types
            def find_markers(pos_list):
                buy_entries, sell_entries, buy_exits, sell_exits = [], [], [], []
                for i in range(1, len(pos_list)):
                    if pos_list[i-1] == 0 and pos_list[i] == 1:
                        buy_entries.append(i)
                    elif pos_list[i-1] == 0 and pos_list[i] == -1:
                        sell_entries.append(i)
                    elif pos_list[i-1] == 1 and pos_list[i] == 0:
                        buy_exits.append(i)
                    elif pos_list[i-1] == -1 and pos_list[i] == 0:
                        sell_exits.append(i)
                return buy_entries, sell_entries, buy_exits, sell_exits

            # Extract markers for both agents
            markers = {}
            for agent_id in [0, 1]:
                be, se, b_ex, s_ex = find_markers(episode_data['positions'][agent_id])
                markers[agent_id] = {
                    'buy_entries': be,
                    'sell_entries': se,
                    'buy_exits': b_ex,
                    'sell_exits': s_ex
                }

            def make_marker_array(indices):
                arr = np.full(len(ohlc_df), np.nan)
                for i in indices:
                    arr[i] = episode_data['prices'][i]
                return arr

            # ---------- Combined chart (both agents) ----------
            ap_combined = []

            # Agent0 markers (larger, bright colours)
            if markers[0]['buy_entries']:
                ap_combined.append(mpf.make_addplot(make_marker_array(markers[0]['buy_entries']),
                                                    type='scatter', marker='^', color='lime', markersize=100, panel=0))
            if markers[0]['sell_entries']:
                ap_combined.append(mpf.make_addplot(make_marker_array(markers[0]['sell_entries']),
                                                    type='scatter', marker='v', color='red', markersize=100, panel=0))
            if markers[0]['buy_exits']:
                ap_combined.append(mpf.make_addplot(make_marker_array(markers[0]['buy_exits']),
                                                    type='scatter', marker='o', color='blue', markersize=80, panel=0))
            if markers[0]['sell_exits']:
                ap_combined.append(mpf.make_addplot(make_marker_array(markers[0]['sell_exits']),
                                                    type='scatter', marker='o', color='orange', markersize=80, panel=0))

            # Agent1 markers (slightly smaller, different colours to distinguish)
            if markers[1]['buy_entries']:
                ap_combined.append(mpf.make_addplot(make_marker_array(markers[1]['buy_entries']),
                                                    type='scatter', marker='^', color='cyan', markersize=80, panel=0))
            if markers[1]['sell_entries']:
                ap_combined.append(mpf.make_addplot(make_marker_array(markers[1]['sell_entries']),
                                                    type='scatter', marker='v', color='magenta', markersize=80, panel=0))
            if markers[1]['buy_exits']:
                ap_combined.append(mpf.make_addplot(make_marker_array(markers[1]['buy_exits']),
                                                    type='scatter', marker='o', color='darkblue', markersize=60, panel=0))
            if markers[1]['sell_exits']:
                ap_combined.append(mpf.make_addplot(make_marker_array(markers[1]['sell_exits']),
                                                    type='scatter', marker='o', color='gold', markersize=60, panel=0))

            fig, axes = mpf.plot(ohlc_df, type='candle', style='charles',
                                 addplot=ap_combined if ap_combined else None, volume=False, returnfig=True,
                                 title=title + ' - Combined')
            plt.savefig('Data/visulizaation/competitive_candlestick_combined.png', dpi=150, bbox_inches='tight')
            plt.show()

            # ---------- Separate charts per agent ----------
            agent_names = {0: 'Agent 0', 1: 'Agent 1'}
            for agent_id in [0, 1]:
                ap_agent = []
                m = markers[agent_id]
                # For single‑agent charts, use the same colours as the original environment
                if m['buy_entries']:
                    ap_agent.append(mpf.make_addplot(make_marker_array(m['buy_entries']), type='scatter', marker='^',
                                                     color='lime', markersize=100, panel=0))
                if m['sell_entries']:
                    ap_agent.append(mpf.make_addplot(make_marker_array(m['sell_entries']), type='scatter', marker='v',
                                                     color='red', markersize=100, panel=0))
                if m['buy_exits']:
                    ap_agent.append(mpf.make_addplot(make_marker_array(m['buy_exits']), type='scatter', marker='o',
                                                     color='blue', markersize=80, panel=0))
                if m['sell_exits']:
                    ap_agent.append(mpf.make_addplot(make_marker_array(m['sell_exits']), type='scatter', marker='o',
                                                     color='orange', markersize=80, panel=0))

                fig_a, axes_a = mpf.plot(ohlc_df, type='candle', style='charles',
                                         addplot=ap_agent if ap_agent else None, volume=False, returnfig=True,
                                         title=f'{title} - {agent_names[agent_id]} Trades')
                plt.savefig(f'Data/visulizaation/competitive_candlestick_agent{agent_id}.png', dpi=150, bbox_inches='tight')
                plt.show()

            # ---------- PnL curves ----------
            fig2, ax = plt.subplots(figsize=(12,6))
            ax.plot(episode_data['steps'], episode_data['pnl'][0], color='green', label='Agent0 PnL')
            ax.plot(episode_data['steps'], episode_data['pnl'][1], color='blue', label='Agent1 PnL')
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_title('PnL Comparison')
            ax.set_xlabel('Step')
            ax.set_ylabel('Total PnL')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('Data/visulizaation/competitive_pnl.png', dpi=150, bbox_inches='tight')
            plt.show()

        except ImportError:
            print("mplfinance not installed. Falling back to simple PnL plot.")
            plt.figure(figsize=(12,6))
            plt.plot(episode_data['steps'], episode_data['pnl'][0], label='Agent0 PnL')
            plt.plot(episode_data['steps'], episode_data['pnl'][1], label='Agent1 PnL')
            plt.legend()
            plt.title('PnL Comparison')
            plt.savefig('Data/visulizaation/competitive_pnl.png')
            plt.show()


# ==========================================================
# Quick test with random actions + visualisation
# ==========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing CompetitiveTradingEnv with random actions")
    print("=" * 60)

    # Create base environment on a short date range
    base_env = TradingEnvironment(start_date="2024-01-01", end_date="2026-02-01")
    comp_env = CompetitiveTradingEnv(base_env,
                                     reward_mode='zero_sum',
                                     include_opponent_position=True,
                                     transaction_cost=0.001)

    print(f"Observation dim per agent: {comp_env.obs_dim}")
    print(f"Market feature dim: {comp_env.market_dim}")
    print(f"Include opponent position: {comp_env.include_opponent_position}")

    obs0, obs1 = comp_env.reset()
    total_rewards = [0.0, 0.0]
    max_steps = min(500, base_env.max_steps - base_env.lookback_window)

    # Data recording for visualisation
    episode_data = {
        'steps': [],
        'time': [],
        'prices': [],
        'open': [], 'high': [], 'low': [],
        'positions': {0: [], 1: []},
        'actions': {0: [], 1: []},
        'pnl': {0: [], 1: []},
    }

    for step in range(max_steps):
        # Random valid actions
        action0 = np.random.choice([-1, 0, 1])
        action1 = np.random.choice([-1, 0, 1])

        obs0, obs1, (rew0, rew1), done, info = comp_env.step((action0, action1))
        total_rewards[0] += rew0
        total_rewards[1] += rew1

        # Record step data (same style as TradingEnvironment)
        row = base_env.data.iloc[base_env.current_step]
        cur_price = row['close']
        episode_data['steps'].append(step)
        episode_data['time'].append(row['time'])
        episode_data['prices'].append(cur_price)
        episode_data['open'].append(row['open'])
        episode_data['high'].append(row['high'])
        episode_data['low'].append(row['low'])
        episode_data['positions'][0].append(comp_env.positions[0])
        episode_data['positions'][1].append(comp_env.positions[1])
        episode_data['actions'][0].append(action0)
        episode_data['actions'][1].append(action1)
        episode_data['pnl'][0].append(comp_env.pnls[0])
        episode_data['pnl'][1].append(comp_env.pnls[1])

        if step % 100 == 0 or done:
            print(f"Step {step:4d}: PnL0={info['pnl_agent0']:8.2f}, "
                  f"PnL1={info['pnl_agent1']:8.2f}, "
                  f"Pos0={comp_env.positions[0]}, Pos1={comp_env.positions[1]}")

        if done:
            break

    print(f"\nFinal PnL Agent 0: {comp_env.pnls[0]:.2f}")
    print(f"Final PnL Agent 1: {comp_env.pnls[1]:.2f}")
    print(f"Total Reward Agent 0: {total_rewards[0]:.2f}")
    print(f"Total Reward Agent 1: {total_rewards[1]:.2f}")
    print(f"Trades Agent 0: {len(info['trade_history_agent0'])}")
    print(f"Trades Agent 1: {len(info['trade_history_agent1'])}")

    # Call the built‑in visualisation
    CompetitiveTradingEnv.visualize_episode(episode_data, title="Competitive Random Test")
    print("✅ Environment test and visualisation completed.")