import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

ACTION_MAP = {
    0: {'name': 'BUY', 'symbol': 'triangle-up', 'color': 'green'},
    1: {'name': 'SELL', 'symbol': 'triangle-down', 'color': 'red'},
    2: {'name': 'HOLD', 'symbol': 'circle', 'color': 'gray'},
    3: {'name': 'NEUTRAL', 'symbol': 'diamond', 'color': 'orange'}
}

def visualize_trading_actions(df, title="Trading Actions", show_actions=True, show_emas=True,
                              fast_ma_col='fast_ma', slow_ma_col='slow_ma'):
    """Create trading actions visualization"""
    
    if not np.issubdtype(df['time'].dtype, np.datetime64):
        df['time'] = pd.to_datetime(df['time'])
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price Chart', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(x=df['time'], open=df['open'], high=df['high'],
                       low=df['low'], close=df['close'], name='Price'),
        row=1, col=1
    )
    
    # EMAs
    if show_emas and fast_ma_col in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time'], y=df[fast_ma_col],
                       name='EMA (fast)', line=dict(color='blue', width=2)),
            row=1, col=1
        )
    if show_emas and slow_ma_col in df.columns:
        fig.add_trace(
            go.Scatter(x=df['time'], y=df[slow_ma_col],
                       name='EMA (slow)', line=dict(color='red', width=2)),
            row=1, col=1
        )
    
    # Action markers
    if show_actions and 'action' in df.columns:
        for action_code, props in ACTION_MAP.items():
            sub = df[df['action'] == action_code]
            if len(sub) == 0:
                continue
            if action_code == 0:      # BUY
                y_pos = sub['low'] * 0.998
            elif action_code == 1:    # SELL
                y_pos = sub['high'] * 1.002
            elif action_code == 3:    # NEUTRAL (close)
                y_pos = sub['close']
            else:                     # HOLD
                y_pos = sub['close']
            
            size = 12 if action_code != 2 else 5
            fig.add_trace(
                go.Scatter(
                    x=sub['time'], y=y_pos, mode='markers',
                    name=props['name'],
                    marker=dict(symbol=props['symbol'], size=size, color=props['color'],
                                line=dict(width=1, color='black')),
                    text=[f"{props['name']}<br>Price: {c:.2f}<br>Time: {t}"
                          for c, t in zip(sub['close'], sub['time'])],
                    hoverinfo='text'
                ),
                row=1, col=1
            )
    
    # Volume
    if 'volume' in df.columns:
        colors = ['green' if c >= o else 'red' for c, o in zip(df['close'], df['open'])]
        fig.add_trace(
            go.Bar(x=df['time'], y=df['volume'], name='Volume',
                   marker_color=colors, opacity=0.5),
            row=2, col=1
        )
    
    fig.update_layout(
        title=title, xaxis_title='Time', yaxis_title='Price',
        template='plotly_dark', hovermode='x unified',
        height=700, width=1400,
        legend=dict(orientation='h', y=1.12)
    )
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    return fig


def visualize_backtest(pnl_curve, preds, positions, df_test=None, action_names=None):
    """Create backtest visualization"""
    if action_names is None:
        action_names = {0:'BUY',1:'SELL',2:'HOLD',3:'NEUTRAL'}
    
    rows = 4 if df_test is not None else 3
    subplot_titles = ['Price & Trade Lines', 'Cumulative P&L', 'Action Distribution', 'Position']
    if df_test is None:
        subplot_titles = ['Cumulative P&L', 'Action Distribution', 'Position']
        rows = 3
    
    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        row_heights=[0.35, 0.22, 0.22, 0.21] if df_test is not None else [0.4, 0.3, 0.3]
    )
    
    # Panel 1: Price with trade lines (if df_test given)
    if df_test is not None:
        df_test = df_test.copy()
        if not np.issubdtype(df_test['time'].dtype, np.datetime64):
            df_test['time'] = pd.to_datetime(df_test['time'])
        
        time_str = df_test['time'].dt.strftime('%Y-%m-%d %H:%M')
        
        fig.add_trace(
            go.Candlestick(x=time_str, open=df_test['open'], high=df_test['high'],
                           low=df_test['low'], close=df_test['close'], name='Price'),
            row=1, col=1
        )
        
        # Action markers
        for action_code, props in [(0, {'name':'BUY','symbol':'triangle-up','color':'green'}),
                                   (1, {'name':'SELL','symbol':'triangle-down','color':'red'}),
                                   (3, {'name':'CLOSE','symbol':'diamond','color':'orange'})]:
            mask = preds == action_code
            if mask.sum() == 0:
                continue
            times_str = time_str.iloc[mask]
            if action_code == 0:
                y_vals = df_test['low'].iloc[mask] * 0.998
            elif action_code == 1:
                y_vals = df_test['high'].iloc[mask] * 1.002
            else:
                y_vals = df_test['close'].iloc[mask]
            fig.add_trace(
                go.Scatter(x=times_str, y=y_vals, mode='markers',
                           name=props['name'],
                           marker=dict(symbol=props['symbol'], size=10, color=props['color'])),
                row=1, col=1
            )
    
    # Panel 2 (or 1): Cumulative P&L
    pnl_row = 2 if df_test is not None else 1
    fig.add_trace(
        go.Scatter(y=pnl_curve, mode='lines', name='Total P&L',
                   line=dict(color='gold', width=2)),
        row=pnl_row, col=1
    )
    
    # Panel 3 (or 2): Action Distribution
    dist_row = 3 if df_test is not None else 2
    actions, counts = np.unique(preds, return_counts=True)
    colors = ['blue','red','green','orange']
    fig.add_trace(
        go.Bar(x=[action_names[a] for a in actions], y=counts,
               marker_color=[colors[a%len(colors)] for a in actions],
               name='Actions'),
        row=dist_row, col=1
    )
    
    # Panel 4 (or 3): Position
    pos_row = 4 if df_test is not None else 3
    fig.add_trace(
        go.Scatter(y=positions, mode='lines', name='Position',
                   line=dict(color='cyan', width=1)),
        row=pos_row, col=1
    )
    
    fig.update_layout(
        title='Backtest Results – Position‑Aware Policy',
        height=300 * rows,
        template='plotly_dark',
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=60)
    )
    
    for i in range(1, rows + 1):
        fig.update_xaxes(rangeslider=dict(visible=False), row=i, col=1)
    
    return fig


def create_live_dashboard(env_state: Dict, predictions: np.ndarray):
    """Create live trading dashboard"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Price & Actions', 'Cumulative P&L', 
                       'Position', 'Action Distribution',
                       'Risk Metrics', 'Trade History'),
        specs=[[{"colspan": 2}, None],
               [{}, {}],
               [{}, {}]],
        vertical_spacing=0.1
    )
    
    # Price chart with actions
    if 'prices' in env_state:
        fig.add_trace(
            go.Scatter(y=env_state['prices'], name='Price',
                      line=dict(color='white', width=1)),
            row=1, col=1
        )
    
    # P&L curve
    if 'pnl_curve' in env_state:
        fig.add_trace(
            go.Scatter(y=env_state['pnl_curve'], name='Total P&L',
                      fill='tozeroy', line=dict(color='gold', width=2)),
            row=2, col=1
        )
    
    # Position
    if 'positions' in env_state:
        fig.add_trace(
            go.Scatter(y=env_state['positions'], name='Position',
                      line=dict(color='cyan', width=2)),
            row=2, col=2
        )
    
    # Action distribution
    if 'action_counts' in env_state:
        actions = list(env_state['action_counts'].keys())
        counts = list(env_state['action_counts'].values())
        colors = ['green', 'red', 'gray', 'orange']
        fig.add_trace(
            go.Bar(x=actions, y=counts, marker_color=colors),
            row=3, col=1
        )
    
    # Risk metrics
    if 'risk_metrics' in env_state:
        metrics = env_state['risk_metrics']
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[list(metrics.keys()), list(metrics.values())])
            ),
            row=3, col=2
        )
    
    fig.update_layout(
        title="Live Trading Dashboard",
        height=900,
        template='plotly_dark',
        showlegend=True
    )
    
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING VISUALIZATION UTILS")
    print("=" * 60)
    
    # Generate mock data
    print("\n📊 Generating mock trading data...")
    np.random.seed(42)
    n_points = 100
    
    dates = pd.date_range(start='2025-01-01', periods=n_points, freq='5min')
    prices = 2000 + np.cumsum(np.random.normal(0, 1, n_points))
    
    # Convert to pandas Series for rolling operations
    prices_series = pd.Series(prices)
    
    mock_df = pd.DataFrame({
        'time': dates,
        'open': prices,
        'high': prices + np.random.uniform(0, 2, n_points),
        'low': prices - np.random.uniform(0, 2, n_points),
        'close': prices + np.random.normal(0, 0.5, n_points),
        'volume': np.random.randint(100, 10000, n_points),
        'action': np.random.choice([0, 1, 2, 3], n_points, p=[0.1, 0.1, 0.7, 0.1]),
        'fast_ma': prices_series.rolling(10).mean().values,
        'slow_ma': prices_series.rolling(20).mean().values
    })
    
    # Fill NaN values
    mock_df = mock_df.bfill().ffill()

    print(f"✓ Generated {len(mock_df)} candles")
    print(f"  Action counts: {mock_df['action'].value_counts().to_dict()}")
    print(f"  Date range: {mock_df['time'].min()} to {mock_df['time'].max()}")
    print(f"  Price range: ${mock_df['close'].min():.2f} - ${mock_df['close'].max():.2f}")
    
    # Test 1: Create trading actions chart
    print("\n📈 Test 1: Creating trading actions chart...")
    try:
        fig = visualize_trading_actions(
            mock_df,
            title="Test Trading Chart",
            show_actions=True,
            show_emas=True
        )
        print("✓ Trading actions chart created")
        print(f"  Figure has {len(fig.data)} traces")
    except Exception as e:
        print(f"✗ Failed to create chart: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Create backtest visualization
    print("\n📉 Test 2: Creating backtest visualization...")
    
    # Mock backtest results
    pnl_curve = np.cumsum(np.random.normal(10, 20, n_points))
    preds = np.random.choice([0, 1, 2, 3], n_points, p=[0.1, 0.1, 0.7, 0.1])
    positions = np.zeros(n_points)
    pos = 0
    for i in range(n_points):
        if preds[i] == 0 and pos == 0:
            pos = 1
        elif preds[i] == 1 and pos == 0:
            pos = -1
        elif preds[i] == 3 and pos != 0:
            pos = 0
        positions[i] = pos
    
    try:
        fig = visualize_backtest(pnl_curve, preds, positions, mock_df)
        print("✓ Backtest visualization created")
    except Exception as e:
        print(f"✗ Failed to create backtest: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Create live dashboard
    print("\n📊 Test 3: Creating live dashboard...")
    
    env_state = {
        'prices': prices[-50:].tolist(),
        'pnl_curve': pnl_curve[-50:].tolist(),
        'positions': positions[-50:].tolist(),
        'actions': preds[-50:].tolist(),
        'action_counts': {0: 10, 1: 8, 2: 70, 3: 12},
        'risk_metrics': {
            'Drawdown': '15.2%',
            'Sharpe': '1.34',
            'Win Rate': '58%',
            'Max Loss': '$250'
        }
    }
    
    try:
        fig = create_live_dashboard(env_state, preds[-10:])
        print("✓ Live dashboard created")
    except Exception as e:
        print(f"✗ Failed to create dashboard: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Test different chart configurations
    print("\n🎨 Test 4: Testing chart configurations...")
    
    configs = [
        {"show_actions": True, "show_emas": True},
        {"show_actions": True, "show_emas": False},
        {"show_actions": False, "show_emas": True},
    ]
    
    for i, config in enumerate(configs, 1):
        try:
            fig = visualize_trading_actions(
                mock_df,
                title=f"Config {i}: actions={config['show_actions']}, emas={config['show_emas']}",
                **config
            )
            print(f"  Config {i}: ✓")
        except Exception as e:
            print(f"  Config {i}: ✗ - {e}")
    
    # Test 5: Test with missing columns
    print("\n⚠️ Test 5: Testing with missing columns...")
    
    minimal_df = mock_df[['time', 'open', 'high', 'low', 'close']].copy()
    try:
        fig = visualize_trading_actions(minimal_df, "Minimal Data Test")
        print("✓ Works with minimal data")
    except Exception as e:
        print(f"✗ Failed with minimal data: {e}")
    
    # Test 6: Test action markers
    print("\n📍 Test 6: Testing action markers...")
    
    for action_code, props in ACTION_MAP.items():
        action_df = mock_df[mock_df['action'] == action_code]
        if len(action_df) > 0:
            print(f"  Action {props['name']}: {len(action_df)} markers")
    
    # Test 7: Test P&L curve generation
    print("\n💰 Test 7: Testing P&L curve...")
    
    # Simulate different trading strategies
    strategies = {
        'Random': np.random.normal(0, 20, 200),
        'Trend': np.cumsum(np.random.normal(2, 5, 200)),
        'Mean Reverting': np.cumsum(np.random.normal(-0.5, 8, 200)),
        'High Risk': np.cumsum(np.random.normal(0, 30, 200))
    }
    
    for name, returns in strategies.items():
        curve = np.cumsum(returns)
        final_pnl = curve[-1]
        max_dd = np.max(np.maximum.accumulate(curve) - curve)
        print(f"  {name:12} - Final: ${final_pnl:.0f}, Max DD: ${max_dd:.0f}")
    
    # Test 8: HTML export test
    print("\n💾 Test 8: Testing HTML export...")
    
    try:
        fig = visualize_trading_actions(mock_df, "Export Test")
        
        # Test if we can save (without actually saving in test)
        html_str = fig.to_html()
        print(f"  HTML generated: {len(html_str):,} characters")
        print("✓ HTML export working")
    except Exception as e:
        print(f"✗ HTML export failed: {e}")
    
    # Test 9: Save sample charts (optional)
    print("\n💾 Test 9: Saving sample charts...")
    try:
        fig = visualize_trading_actions(mock_df, "Sample Trading Chart")
        fig.write_html("Data/visulizaation/test_trading_chart.html")
        print("✓ Saved test_trading_chart.html")
        
        fig = visualize_backtest(pnl_curve, preds, positions, mock_df)
        fig.write_html("Data/visulizaation/test_backtest_chart.html")
        print("✓ Saved test_backtest_chart.html")
        
        fig = create_live_dashboard(env_state, preds[-10:])
        fig.write_html("Data/visulizaation/test_live_dashboard.html")
        print("✓ Saved test_live_dashboard.html")
    except Exception as e:
        print(f"✗ Failed to save charts: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Visualization tests completed!")
    print("=" * 60)
    
    print("\n💡 Note: Sample charts saved as HTML files.")
    print("   Open them in your browser to view the visualizations.")