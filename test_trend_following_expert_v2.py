#!/usr/bin/env python3
"""
Test script: Trend Following Expert with Modified Trading Environment
Optimized for 5-min charts with proper trend detection
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from environments.trading_env import TradingEnvironment
from experts.trend_following import TrendFollowingExpert


def main():
    print("=" * 100)
    print("TREND FOLLOWING EXPERT - 5-MIN CHART OPTIMIZATION")
    print("=" * 100)
    
    # Initialize environment
    print("\n[1/5] Initializing Trading Environment...")
    env = TradingEnvironment()
    obs, info = env.reset()
    
    print(f"✓ Environment initialized")
    print(f"  - Initial Balance: ${info['balance']:.2f}")
    print(f"  - Data size: {len(env.data)} candles")
    print(f"  - Action space: Discrete(3) - BUY(0), SELL(1), HOLD(2)")
    
    # Initialize expert
    print("\n[2/5] Initializing Trend Following Expert...")
    expert = TrendFollowingExpert(fast_ma=5, slow_ma=10)  # Optimized for 5-min
    expert.reset_position()
    print(f"✓ Expert initialized for 5-min charts")
    print(f"  - Fast MA period: 5 candles (~25 min)")
    print(f"  - Slow MA period: 10 candles (~50 min)")
    print(f"  - RSI period: 9 candles")
    
    # Prepare data with technical indicators
    print("\n[3/5] Calculating Technical Indicators...")
    env.data = expert.prepare_data_with_indicators(env.data)
    print(f"✓ Added indicators to data:")
    print(f"  - fast_ma: Fast moving average")
    print(f"  - slow_ma: Slow moving average")
    print(f"  - RSI: Relative Strength Index")
    print(f"  - momentum: Price momentum")
    print(f"  - hl_range: High-Low range")
    
    # Run simulation
    print("\n[4/5] Running Expert Trading Simulation...")
    print("=" * 100)
    
    episode_data = {
        'step': [],
        'time': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'positions': [],
        'actions': [],
        'action_names': [],
        'balance': [],
        'total_pnl': [],
        'trade_executed': [],
        'entry_prices': [],
        'trade_types': [],
        'signal_strength': [],
    }
    
    step_count = 0
    trade_count = 0
    buy_signals = 0
    sell_signals = 0
    hold_signals = 0
    
    for step_num in range(500):
        # Get expert action based on current state
        expert_action = expert.get_expert_action(
            env.data, 
            env.current_step,
            current_position=env.current_position
        )
        
        # Count signal types
        if expert_action == 0:
            buy_signals += 1
        elif expert_action == 1:
            sell_signals += 1
        else:
            hold_signals += 1
        
        # Execute action in environment
        obs, reward, terminated, truncated, info = env.step(expert_action)
        
        # Collect data
        current_price = env._get_current_price()
        row = env.data.iloc[env.current_step - 1] if env.current_step > 0 else env.data.iloc[0]
        
        episode_data['step'].append(step_num)
        episode_data['time'].append(row.get('time', step_num) if 'time' in row else step_num)
        episode_data['open'].append(row.get('open', current_price))
        episode_data['high'].append(row.get('high', current_price))
        episode_data['low'].append(row.get('low', current_price))
        episode_data['close'].append(current_price)
        episode_data['positions'].append(info['position'])
        episode_data['actions'].append(expert_action)
        episode_data['action_names'].append(info['action_name'])
        episode_data['balance'].append(info['balance'])
        episode_data['total_pnl'].append(info['total_pnl'])
        episode_data['trade_executed'].append(info['trade_executed'])
        episode_data['entry_prices'].append(env.entry_price if env.current_position != 0 else None)
        
        if info['trade_executed']:
            if expert_action == 0 and env.current_position == 1:
                episode_data['trade_types'].append('LONG_ENTRY')
            elif expert_action == 1 and env.current_position == -1:
                episode_data['trade_types'].append('SHORT_ENTRY')
            elif expert_action == 0 and env.current_position == 0:
                episode_data['trade_types'].append('SHORT_EXIT')
            elif expert_action == 1 and env.current_position == 0:
                episode_data['trade_types'].append('LONG_EXIT')
            else:
                episode_data['trade_types'].append('TRADE')
            trade_count += 1
        else:
            episode_data['trade_types'].append(None)
        
        # Get indicator values for debugging
        ma_diff = row.get('fast_ma', 0) - row.get('slow_ma', 0) if 'fast_ma' in row and 'slow_ma' in row else 0
        episode_data['signal_strength'].append(ma_diff)
        
        step_count = step_num
        
        # Print progress with signal details
        if step_num % 100 == 0:
            pos_str = 'LONG ' if info['position'] == 1 else 'SHORT' if info['position'] == -1 else 'FLAT '
            ma_trend = "↑" if ma_diff > 0 else "↓" if ma_diff < 0 else "-"
            rsi_val = row.get('RSI', 50) if 'RSI' in row else 50
            print(f"  Step {step_num:4d}: {info['action_name']:4s} | Price: {current_price:8.2f} | "
                  f"Pos: {pos_str} | Balance: ${info['balance']:10.2f} | PnL: ${info['total_pnl']:8.2f} | "
                  f"MA: {ma_trend} | RSI: {rsi_val:.1f}")
        
        if terminated:
            print(f"\n  ✓ Episode terminated at step {step_num}")
            break
    
    # Print signal statistics
    print("\n" + "=" * 100)
    print("Signal Statistics:")
    print(f"  BUY signals:  {buy_signals:4d} ({buy_signals/step_count*100:5.1f}%)")
    print(f"  SELL signals: {sell_signals:4d} ({sell_signals/step_count*100:5.1f}%)")
    print(f"  HOLD signals: {hold_signals:4d} ({hold_signals/step_count*100:5.1f}%)")
    
    # Print results
    print("\n[5/5] Simulation Complete - Summary Statistics")
    print("=" * 100)
    print(f"Simulation Steps: {step_count + 1}")
    print(f"Final Balance: ${info['balance']:.2f}")
    print(f"Initial Balance: ${env.initial_balance:.2f}")
    print(f"Total PnL: ${info['total_pnl']:.2f}")
    print(f"Realized PnL: ${info['realized_pnl']:.2f}")
    print(f"Unrealized PnL: ${info['unrealized_pnl']:.2f}")
    print(f"Return: {((info['balance'] - env.initial_balance) / env.initial_balance * 100):.2f}%")
    print(f"\nTrades Completed: {len(info['trade_history'])}")
    print(f"Trade Signals Generated: {trade_count}")
    
    if len(info['trade_history']) > 0:
        wins = sum(1 for t in info['trade_history'] if t['pnl'] > 0)
        losses = sum(1 for t in info['trade_history'] if t['pnl'] < 0)
        print(f"Winning Trades: {wins}")
        print(f"Losing Trades: {losses}")
        print(f"Win Rate: {wins / len(info['trade_history']) * 100:.2f}%")
        avg_win = np.mean([t['pnl'] for t in info['trade_history'] if t['pnl'] > 0]) if wins > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in info['trade_history'] if t['pnl'] < 0]) if losses > 0 else 0
        print(f"Avg Win: ${avg_win:.2f}")
        print(f"Avg Loss: ${avg_loss:.2f}")
        if losses > 0:
            print(f"Profit Factor: {abs(wins * avg_win / (losses * avg_loss)):.2f}x" if avg_loss != 0 else "Infinite")
    
    # Create visualizations
    print("\n" + "=" * 100)
    print("Generating Visualizations...")
    print("=" * 100)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Rectangle
        
        # Convert to DataFrame for easier handling
        df_episode = pd.DataFrame(episode_data)
        
        # Create figure with candlestick and indicators
        fig, axes = plt.subplots(4, 1, figsize=(20, 14), 
                                  gridspec_kw={'height_ratios': [2.5, 1, 1, 1]})
        
        # ============ PLOT 1: Candlestick Chart with Trade Signals ============
        ax1 = axes[0]
        
        # Draw candlesticks
        width = 0.6
        width2 = 0.05
        
        for idx, row in df_episode.iterrows():
            open_p = row['open']
            close_p = row['close']
            high_p = row['high']
            low_p = row['low']
            
            # Color based on close vs open
            if close_p >= open_p:
                color = 'green'
                body_height = close_p - open_p
                body_bottom = open_p
            else:
                color = 'red'
                body_height = open_p - close_p
                body_bottom = close_p
            
            # Draw high-low line (wick)
            ax1.plot([idx, idx], [low_p, high_p], color=color, linewidth=1, alpha=0.6)
            
            # Draw open-close box (body)
            rect = Rectangle((idx - width2, body_bottom), width2 * 2, body_height,
                           facecolor=color, edgecolor=color, linewidth=1, alpha=0.8)
            ax1.add_patch(rect)
        
        # Mark trade entry points
        long_entries = df_episode[df_episode['trade_types'] == 'LONG_ENTRY']
        short_entries = df_episode[df_episode['trade_types'] == 'SHORT_ENTRY']
        long_exits = df_episode[df_episode['trade_types'] == 'LONG_EXIT']
        short_exits = df_episode[df_episode['trade_types'] == 'SHORT_EXIT']
        
        if len(long_entries) > 0:
            ax1.scatter(long_entries.index, long_entries['close'], 
                       marker='^', color='lime', s=200, label='BUY Entry', 
                       zorder=5, edgecolors='darkgreen', linewidth=2)
        
        if len(short_entries) > 0:
            ax1.scatter(short_entries.index, short_entries['close'],
                       marker='v', color='crimson', s=200, label='SELL Entry',
                       zorder=5, edgecolors='darkred', linewidth=2)
        
        if len(long_exits) > 0:
            ax1.scatter(long_exits.index, long_exits['close'],
                       marker='X', color='orange', s=250, label='LONG Exit',
                       zorder=5, edgecolors='darkorange', linewidth=2)
        
        if len(short_exits) > 0:
            ax1.scatter(short_exits.index, short_exits['close'],
                       marker='X', color='purple', s=250, label='SHORT Exit',
                       zorder=5, edgecolors='indigo', linewidth=2)
        
        # Add price line
        ax1.plot(df_episode.index, df_episode['close'], color='black', 
                linewidth=0.5, alpha=0.3, linestyle='--')
        
        ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax1.set_title('5-Min Candlestick Chart - Trend Following Expert Signals', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(left=0)
        
        # ============ PLOT 2: Moving Averages ============
        ax2 = axes[1]
        if 'fast_ma' in env.data.columns and 'slow_ma' in env.data.columns:
            fast_ma_vals = env.data['fast_ma'].iloc[env.lookback_window:env.lookback_window+len(df_episode)]
            slow_ma_vals = env.data['slow_ma'].iloc[env.lookback_window:env.lookback_window+len(df_episode)]
            
            ax2.plot(df_episode.index, fast_ma_vals, color='blue', linewidth=2, label='Fast MA (5)')
            ax2.plot(df_episode.index, slow_ma_vals, color='red', linewidth=2, label='Slow MA (10)')
            ax2.plot(df_episode.index, df_episode['close'], color='black', linewidth=0.5, alpha=0.3, linestyle='--')
            ax2.set_ylabel('Moving Averages', fontsize=11, fontweight='bold')
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(left=0)
        
        # ============ PLOT 3: Account Balance ============
        ax3 = axes[2]
        ax3.plot(df_episode.index, df_episode['balance'], color='darkblue', 
                linewidth=2.5, label='Balance', zorder=2)
        ax3.axhline(y=env.initial_balance, color='gray', linestyle='--',
                   label='Initial Balance', linewidth=1.5, alpha=0.7)
        ax3.fill_between(df_episode.index, env.initial_balance, df_episode['balance'],
                        alpha=0.2, color='darkblue')
        
        ax3.set_ylabel('Balance ($)', fontsize=11, fontweight='bold')
        ax3.set_title('Account Balance Over Time', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xlim(left=0)
        
        # ============ PLOT 4: Position State ============
        ax4 = axes[3]
        colors = ['red' if p == -1 else 'lime' if p == 1 else 'lightgray' 
                 for p in df_episode['positions']]
        ax4.bar(df_episode.index, df_episode['positions'], color=colors,
               width=1, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax4.set_xlabel('Step', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Position', fontsize=11, fontweight='bold')
        ax4.set_yticks([-1, 0, 1])
        ax4.set_yticklabels(['SHORT (-1)', 'FLAT (0)', 'LONG (+1)'])
        ax4.set_title('Position State - Expert Trend Signals', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax4.set_xlim(left=0)
        
        # Add legend for position colors
        short_patch = mpatches.Patch(color='red', label='SHORT Position', alpha=0.8)
        long_patch = mpatches.Patch(color='lime', label='LONG Position', alpha=0.8)
        flat_patch = mpatches.Patch(color='lightgray', label='FLAT (No Position)', alpha=0.8)
        ax4.legend(handles=[long_patch, short_patch, flat_patch], loc='best', fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path('Data/visulizaation/trend_following_candlestick_chart.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Candlestick chart saved: {output_path}")
        plt.close()
        
        # Create detailed trade analysis figure
        create_trade_analysis_chart(info['trade_history'], env.initial_balance)
        
    except ImportError as e:
        print(f"⚠ Could not generate visualization: {e}")
    
    print("\n" + "=" * 100)
    print("✅ Test completed successfully!")
    print("=" * 100)


def create_trade_analysis_chart(trade_history, initial_balance):
    """Create detailed trade analysis chart"""
    try:
        import matplotlib.pyplot as plt
        
        if not trade_history:
            print("⚠ No trades to analyze")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract trade data
        pnls = [t['pnl'] for t in trade_history]
        types = [t['type'] for t in trade_history]
        
        # Plot 1: PnL distribution
        ax1 = axes[0, 0]
        colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
        ax1.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Trade Number', fontweight='bold')
        ax1.set_ylabel('PnL ($)', fontweight='bold')
        ax1.set_title('Individual Trade PnL Distribution', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Cumulative PnL
        ax2 = axes[0, 1]
        cumulative_pnl = np.cumsum(pnls)
        ax2.plot(cumulative_pnl, color='darkblue', linewidth=2, marker='o', markersize=5)
        ax2.fill_between(range(len(cumulative_pnl)), cumulative_pnl, alpha=0.3, color='darkblue')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Trade Number', fontweight='bold')
        ax2.set_ylabel('Cumulative PnL ($)', fontweight='bold')
        ax2.set_title('Cumulative Profit/Loss', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trade type distribution
        ax3 = axes[1, 0]
        long_count = types.count('LONG')
        short_count = types.count('SHORT')
        ax3.bar(['LONG', 'SHORT'], [long_count, short_count], color=['lime', 'crimson'], alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Number of Trades', fontweight='bold')
        ax3.set_title('Trade Type Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Statistics text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        wins = sum(1 for pnl in pnls if pnl > 0)
        losses = sum(1 for pnl in pnls if pnl < 0)
        breakeven = sum(1 for pnl in pnls if pnl == 0)
        
        stats_text = f"""
TRADE STATISTICS - 5-MIN CHART

Total Trades: {len(trade_history)}
Winning Trades: {wins}
Losing Trades: {losses}
Breakeven: {breakeven}

Win Rate: {wins/len(trade_history)*100:.2f}%
Loss Rate: {losses/len(trade_history)*100:.2f}%

Total PnL: ${sum(pnls):.2f}
Average PnL/Trade: ${np.mean(pnls):.2f}
Max Win: ${max(pnls):.2f}
Max Loss: ${min(pnls):.2f}
Std Dev: ${np.std(pnls):.2f}

Profit Factor: {abs(sum(p for p in pnls if p > 0) / sum(p for p in pnls if p < 0)) if any(p < 0 for p in pnls) else 'N/A'}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        output_path = Path('Data/visulizaation/trade_analysis_statistics.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Trade analysis saved: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"⚠ Could not create trade analysis: {e}")


if __name__ == "__main__":
    main()
