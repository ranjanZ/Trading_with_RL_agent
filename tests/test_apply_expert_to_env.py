import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import os
from tqdm import tqdm  # progress bar
from matplotlib.dates import DateFormatter
from utils.metrics import calculate_trading_metrics


def run_expert_and_plot(
    env: 'TradingEnvironment',
    expert: 'TrendFollowingExpert',
    max_steps: Optional[int] = None,
    plot_dir: str = "Data/visulizaation"
):
    """
    Run the expert strategy on the environment, collect data, and create plots.
    """
    # 1. Prepare expert indicators
    env.data = expert.prepare_data_with_indicators(env.data)

    # 2. Reset environment and expert
    obs, info = env.reset()
    expert.reset_position()
    expert.current_position = env.current_position

    total_steps = max_steps if max_steps is not None else env.max_steps - env.lookback_window

    # 3. Data collection containers
    episode_data = {
        'steps': [],
        'time': [],
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

    # 4. Run episode with progress bar
    print("Running expert strategy...")
    for step in tqdm(range(total_steps), desc="Steps", unit="step"):
        action = expert.get_expert_action(
            env.data,
            env.current_step,
            current_position=env.current_position
        )
        obs, reward, terminated, truncated, info = env.step(action)
        expert.current_position = env.current_position

        row = env.data.iloc[env.current_step]
        current_price = row['close']
        episode_data['steps'].append(step)
        episode_data['time'].append(row['time'])
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
            break

    # 5. Print summary & date range
    print(f"\nExpert run finished. Final balance: {info['balance']:.2f}")
    print(f"Total PnL: {info['total_pnl']:.2f}")
    print(f"Trades executed: {len(info['trade_history'])}")

    # Date range
    start_date = pd.to_datetime(episode_data['time'][0])
    end_date = pd.to_datetime(episode_data['time'][-1])
    print(f"\nDate range: {start_date}  →  {end_date}")

    # Trading metrics
    trading_metrics = calculate_trading_metrics(info['trade_history'])
    print("\n--- Trading Metrics ---")
    for key, value in trading_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # 6. Produce plots
    _create_plots(episode_data, env.initial_balance, plot_dir, trading_metrics)


def _create_plots(episode_data: dict, initial_balance: float, plot_dir: str, trading_metrics: dict):
    """Build candlestick chart (static PNG) and a 3‑panel balance/price/PnL plot with metrics."""
    import mplfinance as mpf
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.dates import DateFormatter

    os.makedirs(plot_dir, exist_ok=True)

    # ---------- Candlestick chart (with markers) ----------
    ohlc_df = pd.DataFrame({
        'Open':  episode_data['open'],
        'High':  episode_data['high'],
        'Low':   episode_data['low'],
        'Close': episode_data['prices']
    }, index=pd.DatetimeIndex(episode_data['time']))

    # Detect entry/exit points
    pos = episode_data['positions']
    buy_entry = [i for i in range(1, len(pos)) if pos[i-1]==0 and pos[i]==1]
    sell_entry = [i for i in range(1, len(pos)) if pos[i-1]==0 and pos[i]==-1]
    buy_exit = [i for i in range(1, len(pos)) if pos[i-1]==1 and pos[i]==0]
    sell_exit = [i for i in range(1, len(pos)) if pos[i-1]==-1 and pos[i]==0]

    n = len(ohlc_df)
    markers = {
        'buy_entry': np.full(n, np.nan),
        'sell_entry': np.full(n, np.nan),
        'buy_exit': np.full(n, np.nan),
        'sell_exit': np.full(n, np.nan),
    }
    for step in buy_entry:
        markers['buy_entry'][step] = episode_data['prices'][step]
    for step in sell_entry:
        markers['sell_entry'][step] = episode_data['prices'][step]
    for step in buy_exit:
        markers['buy_exit'][step] = episode_data['prices'][step]
    for step in sell_exit:
        markers['sell_exit'][step] = episode_data['prices'][step]

    ap = []
    if buy_entry:
        ap.append(mpf.make_addplot(markers['buy_entry'], type='scatter', marker='^', color='lime', markersize=100, panel=0))
    if sell_entry:
        ap.append(mpf.make_addplot(markers['sell_entry'], type='scatter', marker='v', color='red', markersize=100, panel=0))
    if buy_exit:
        ap.append(mpf.make_addplot(markers['buy_exit'], type='scatter', marker='o', color='blue', markersize=80, panel=0))
    if sell_exit:
        ap.append(mpf.make_addplot(markers['sell_exit'], type='scatter', marker='o', color='orange', markersize=80, panel=0))

    # Plot candlestick – show it, then save
    fig, axes = mpf.plot(ohlc_df,
                         type='candle',
                         style='charles',
                         addplot=ap if ap else None,
                         volume=False,
                         returnfig=True,
                         title='Expert Strategy - Candlestick with Trades')
    plt.savefig(f'{plot_dir}/expert_candlestick.png', dpi=150, bbox_inches='tight')
    plt.show()          # show the candlestick plot
    plt.close(fig)

    # ---------- 3‑panel plot: Price, Balance, PnL ----------
    dates = pd.to_datetime(episode_data['time'])
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Closing price
    ax1.plot(dates, episode_data['prices'], label='Close Price', color='black', linewidth=1)
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Price Movement')

    # Panel 2: Account balance
    ax2.plot(dates, episode_data['balance'], label='Balance', color='blue')
    ax2.axhline(y=initial_balance, color='gray', linestyle='--', label='Initial')
    ax2.set_ylabel('Balance ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: PnL (realized and total)
    ax3.plot(dates, episode_data['total_pnl'], label='Total PnL', color='green')
    ax3.plot(dates, episode_data['realized_pnl'], label='Realized PnL', color='orange')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_xlabel('Date/Time')
    ax3.set_ylabel('PnL ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Format x-axis dates
    for ax in (ax1, ax2, ax3):
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add metrics text box (on the PnL panel)
    metrics_text = (
        f"Total Trades: {trading_metrics.get('total_trades', 0)}\n"
        f"Win Rate: {trading_metrics.get('win_rate', 0)*100:.1f}%\n"
        f"Total PnL: ${trading_metrics.get('total_pnl', 0):.2f}\n"
        f"Profit Factor: {trading_metrics.get('profit_factor', 0):.2f}\n"
        f"Sharpe Ratio: {trading_metrics.get('sharpe_ratio', 0):.2f}\n"
        f"Max Win: ${trading_metrics.get('max_win', 0):.2f}\n"
        f"Max Loss: ${trading_metrics.get('max_loss', 0):.2f}"
    )
    ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/expert_balance_pnl.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig2)


if __name__ == "__main__":
    from environments.trading_env import TradingEnvironment
    from experts.trend_following import TrendFollowingExpert
    from experts.random_expert import RandomExpert  


    env = TradingEnvironment()
    expert = TrendFollowingExpert(fast_ma=5, slow_ma=10)
    #expert=RandomExpert(p_buy=0.33, p_sell=0.33)
    run_expert_and_plot(env, expert, plot_dir="Data/visulizaation")