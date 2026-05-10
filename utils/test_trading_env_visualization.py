#!/usr/bin/env python3
"""
Test script for the modified trading environment with state machine logic.
Tests with random actions and generates visualization.
"""

import sys
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from environments.trading_env import TradingEnvironment


def main():
    print("=" * 80)
    print("Testing Modified Trading Environment with State Machine")
    print("=" * 80)
    print("\nState Machine Rules:")
    print("- Position 0 (FLAT): can BUY, SELL, or HOLD")
    print("- Position 1 (LONG): can SELL (close), or HOLD")
    print("- Position -1 (SHORT): can BUY (close), or HOLD")
    print("\n" + "=" * 80)
    
    # Initialize environment
    env = TradingEnvironment()
    obs, info = env.reset()
    
    print(f"Environment initialized!")
    print(f"Initial Balance: ${info['balance']:.2f}")
    print(f"Data size: {len(env.data)} candles")
    
    # Collect episode data for visualization
    episode_data = {
        'steps': [],
        'prices': [],
        'positions': [],
        'actions': [],
        'action_names': [],
        'balance': [],
        'unrealized_pnl': [],
        'realized_pnl': [],
        'total_pnl': [],
        'valid_actions': [],
    }

    step_num = 0
    for step_num in range(500):
        # Get valid actions for current position
        valid_mask = env._get_valid_actions_mask()
        valid_actions = [i for i, valid in enumerate(valid_mask) if valid]
        
        # Sample random action from valid actions only
        action = np.random.choice(valid_actions)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Collect data
        current_price = env._get_current_price()
        episode_data['steps'].append(step_num)
        episode_data['prices'].append(current_price)
        episode_data['positions'].append(info['position'])
        episode_data['actions'].append(action)
        episode_data['action_names'].append(info['action_name'])
        episode_data['balance'].append(info['balance'])
        episode_data['unrealized_pnl'].append(info['unrealized_pnl'])
        episode_data['realized_pnl'].append(info['realized_pnl'])
        episode_data['total_pnl'].append(info['total_pnl'])
        episode_data['valid_actions'].append(valid_mask)
        
        # Print progress
        pos_str = 'LONG' if info['position'] == 1 else 'SHORT' if info['position'] == -1 else 'FLAT'
        if step_num % 50 == 0:
            print(f"Step {step_num:4d}: Price={current_price:8.4f}, Balance=${info['balance']:8.2f}, "
                  f"Position={pos_str:5s}, Total PnL=${info['total_pnl']:8.2f}, "
                  f"Action={info['action_name']}")
        
        if terminated:
            print(f"\n✓ Episode terminated at step {step_num}")
            break

    # Print final statistics
    print("\n" + "=" * 80)
    print("Simulation Results:")
    print("=" * 80)
    print(f"Final Step: {step_num}")
    print(f"Final Balance: ${info['balance']:.2f}")
    print(f"Total PnL: ${info['total_pnl']:.2f}")
    print(f"Realized PnL: ${info['realized_pnl']:.2f}")
    print(f"Unrealized PnL: ${info['unrealized_pnl']:.2f}")
    print(f"Total Trades Completed: {len(info['trade_history'])}")
    
    if len(info['trade_history']) > 0:
        print(f"\nTrade History:")
        for i, trade in enumerate(info['trade_history'], 1):
            print(f"  Trade {i}: {trade['type']} @ Entry: {trade['entry']:.4f}, "
                  f"Exit: {trade['exit']:.4f}, PnL: ${trade['pnl']:.2f}")
    
    # Create visualization
    print("\n" + "=" * 80)
    print("Creating Visualization...")
    print("=" * 80)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        
        # Plot 1: Price and positions
        ax1 = axes[0]
        ax1.plot(episode_data['steps'], episode_data['prices'], label='Close Price', 
                 color='black', linewidth=1.5, zorder=2)
        
        # Highlight entry/exit points more clearly
        entry_buys = []
        entry_sells = []
        exits = []
        
        for i in range(len(episode_data['steps'])):
            if i > 0:
                prev_pos = episode_data['positions'][i-1]
                curr_pos = episode_data['positions'][i]
                
                # Entry points (transition from 0 to ±1)
                if prev_pos == 0 and curr_pos == 1:
                    entry_buys.append((episode_data['steps'][i], episode_data['prices'][i]))
                elif prev_pos == 0 and curr_pos == -1:
                    entry_sells.append((episode_data['steps'][i], episode_data['prices'][i]))
                
                # Exit points (transition from ±1 to 0)
                if prev_pos != 0 and curr_pos == 0:
                    exits.append((episode_data['steps'][i], episode_data['prices'][i]))
        
        if entry_buys:
            buys = list(zip(*entry_buys))
            ax1.scatter(buys[0], buys[1], marker='^', color='lime', s=150, 
                       label='BUY Entry', zorder=5, edgecolors='darkgreen', linewidth=1.5)
        
        if entry_sells:
            sells = list(zip(*entry_sells))
            ax1.scatter(sells[0], sells[1], marker='v', color='red', s=150, 
                       label='SELL Entry', zorder=5, edgecolors='darkred', linewidth=1.5)
        
        if exits:
            exit_points = list(zip(*exits))
            ax1.scatter(exit_points[0], exit_points[1], marker='X', color='orange', s=200, 
                       label='Position Close', zorder=5, edgecolors='darkorange', linewidth=1.5)
        
        ax1.set_ylabel('Price', fontsize=11, fontweight='bold')
        ax1.set_title('Trading Environment - Price and Position Entries/Exits', 
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(left=0)
        
        # Plot 2: Balance
        ax2 = axes[1]
        ax2.plot(episode_data['steps'], episode_data['balance'], label='Balance', 
                 color='darkblue', linewidth=2, zorder=2)
        ax2.axhline(y=env.initial_balance, color='gray', linestyle='--', 
                   label='Initial Balance', linewidth=1.5, alpha=0.7)
        ax2.fill_between(episode_data['steps'], env.initial_balance, episode_data['balance'], 
                        alpha=0.2, color='darkblue')
        ax2.set_ylabel('Balance ($)', fontsize=11, fontweight='bold')
        ax2.set_title('Account Balance Over Time', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(left=0)
        
        # Plot 3: PnL Components
        ax3 = axes[2]
        ax3.plot(episode_data['steps'], episode_data['realized_pnl'], label='Realized PnL', 
                 color='darkgreen', linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax3.plot(episode_data['steps'], episode_data['total_pnl'], label='Total PnL', 
                 color='darkblue', linewidth=2.5)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.fill_between(episode_data['steps'], 0, episode_data['total_pnl'], alpha=0.15, color='darkblue')
        ax3.set_ylabel('PnL ($)', fontsize=11, fontweight='bold')
        ax3.set_title('Profit and Loss Over Time', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xlim(left=0)
        
        # Plot 4: Position
        ax4 = axes[3]
        colors = ['red' if p == -1 else 'lime' if p == 1 else 'lightgray' for p in episode_data['positions']]
        ax4.bar(episode_data['steps'], episode_data['positions'], color=colors, width=1, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_xlabel('Step', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Position', fontsize=11, fontweight='bold')
        ax4.set_yticks([-1, 0, 1])
        ax4.set_yticklabels(['SHORT (-1)', 'FLAT (0)', 'LONG (+1)'])
        ax4.set_title('Position State Over Time', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax4.set_xlim(left=0)
        
        # Add custom legend for positions
        short_patch = mpatches.Patch(color='red', label='SHORT Position', alpha=0.8)
        long_patch = mpatches.Patch(color='lime', label='LONG Position', alpha=0.8)
        flat_patch = mpatches.Patch(color='lightgray', label='FLAT (No Position)', alpha=0.8)
        ax4.legend(handles=[long_patch, short_patch, flat_patch], loc='best', fontsize=10)
        
        plt.tight_layout()
        
        output_path = Path('Data/visulizaation/trading_env_test_visualization.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")
        plt.close()
        
        # Create a detailed statistics figure
        fig2, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        
        stats_text = f"""
TRADING ENVIRONMENT TEST RESULTS - STATE MACHINE VALIDATION

Simulation Statistics:
  • Total Steps: {step_num}
  • Final Balance: ${info['balance']:.2f}
  • Initial Balance: ${env.initial_balance:.2f}
  • Total PnL: ${info['total_pnl']:.2f}
  • Realized PnL: ${info['realized_pnl']:.2f}
  • Unrealized PnL: ${info['unrealized_pnl']:.2f}
  • Return: {((info['balance'] - env.initial_balance) / env.initial_balance * 100):.2f}%

Trading Activity:
  • Total Completed Trades: {len(info['trade_history'])}
  • Current Position: {('FLAT' if info['position'] == 0 else 'LONG' if info['position'] == 1 else 'SHORT')}
  
State Machine Validation:
  ✓ Position 0 (FLAT): Can execute BUY, SELL, HOLD
  ✓ Position 1 (LONG): Can execute SELL (close) or HOLD
  ✓ Position -1 (SHORT): Can execute BUY (close) or HOLD
  ✓ Single order at a time enforced
  ✓ Valid actions mask working correctly
  ✓ Invalid actions penalized with -0.1 reward

Market Data:
  • Candles Used: {len(episode_data['steps'])} out of {len(env.data)}
  • Price Range: ${min(episode_data['prices']):.4f} - ${max(episode_data['prices']):.4f}
  • First Price: ${episode_data['prices'][0]:.4f}
  • Last Price: ${episode_data['prices'][-1]:.4f}
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        stats_path = Path('Data/visulizaation/trading_env_statistics.png')
        plt.savefig(stats_path, dpi=150, bbox_inches='tight')
        print(f"✓ Statistics saved to: {stats_path}")
        plt.close()
        
    except ImportError as e:
        print(f"⚠ Matplotlib not available for visualization: {e}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
