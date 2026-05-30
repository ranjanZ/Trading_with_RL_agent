import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as   pd


def calculate_imitation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate metrics for imitation learning"""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    # Per-class metrics
    for class_id in range(4):
        precision = precision_score(y_true, y_pred, labels=[class_id], average='micro', zero_division=0)
        recall = recall_score(y_true, y_pred, labels=[class_id], average='micro', zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=[class_id], average='micro', zero_division=0)
        
        metrics[f'precision_class_{class_id}'] = precision
        metrics[f'recall_class_{class_id}'] = recall
        metrics[f'f1_class_{class_id}'] = f1
    
    return metrics


import numpy as np
from typing import Dict, List, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

def calculate_trading_metrics(trades: List[Dict], initial_capital: float = 10000.0,
                              annualize: bool = True, periods_per_year: int = 252) -> Dict:
    """
    Calculate trading metrics, including an annualised Sharpe ratio derived
    from a daily equity curve.

    - If trades contain 'date' or 'timestamp' keys, daily PnL is aggregated
      to form a true daily equity curve.
    - Otherwise, each trade is treated as a "day" (sequential step) to
      produce an equity curve.
    """
    if not trades:
        return {}

    # Basic statistics (unchanged)
    pnls = [t['pnl'] for t in trades]
    winning = [p for p in pnls if p > 0]
    losing  = [p for p in pnls if p < 0]

    metrics = {
        'total_trades': len(trades),
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'win_rate': len(winning) / len(trades) if trades else 0,
        'total_pnl': sum(pnls),
        'avg_win': np.mean(winning) if winning else 0,
        'avg_loss': np.mean(losing) if losing else 0,
        'max_win': max(pnls) if pnls else 0,
        'max_loss': min(pnls) if pnls else 0,
        'profit_factor': (abs(sum(winning) / sum(losing))
                          if losing else float('inf')),
    }

    # ----- Build equity curve -----
    # Try to use timestamps if available and pandas is installed
    equity = None
    if HAS_PANDAS and trades:
        has_date = 'date' in trades[0] or 'timestamp' in trades[0]
        if has_date:
            try:
                date_col = 'date' if 'date' in trades[0] else 'timestamp'
                trade_df = pd.DataFrame(trades)
                trade_df[date_col] = pd.to_datetime(trade_df[date_col])
                daily_pnl = trade_df.groupby(trade_df[date_col].dt.date)['pnl'].sum()
                equity = initial_capital + daily_pnl.cumsum()  # pd.Series
            except Exception:
                equity = None  # fallback to sequential

    if equity is None:
        # No timestamps or pandas unavailable → treat each trade as a "day"
        equity = np.cumsum([initial_capital] + pnls)

    # ----- Daily returns -----
    if isinstance(equity, pd.Series):
        daily_returns = equity.pct_change().dropna().values  # numpy array
    else:
        # equity is a numpy array
        daily_returns = np.diff(equity) / equity[:-1]

    # ----- Sharpe ratio -----
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        daily_sharpe = np.mean(daily_returns) / (np.std(daily_returns, ddof=1) + 1e-10)
        sharpe = daily_sharpe * np.sqrt(periods_per_year) if annualize else daily_sharpe
    else:
        sharpe = 0.0

    metrics['sharpe_ratio'] = sharpe
    # Keep the old simple measure for reference
    metrics['trade_signal_to_noise'] = (
        np.mean(pnls) / (np.std(pnls) + 1e-8) if len(pnls) > 1 else 0.0
    )

    return metrics

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING METRICS CALCULATIONS")
    print("=" * 60)
    
    # Test 1: Calculate imitation metrics
    print("\n📊 Test 1: Testing imitation metrics...")
    
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.1, 0.7, 0.1])
    
    # Generate predictions with different accuracies
    prediction_scenarios = {
        'Perfect': y_true.copy(),
        'Random': np.random.choice([0, 1, 2, 3], n_samples),
        'Biased': np.where(np.random.random(n_samples) < 0.7, y_true, np.random.choice([0, 1, 2, 3], n_samples)),
        'Mostly_HOLD': np.full(n_samples, 2)
    }
    
    for name, y_pred in prediction_scenarios.items():
        metrics = calculate_imitation_metrics(y_true, y_pred)
        print(f"\n  {name}:")
        print(f"    Accuracy: {metrics['accuracy']*100:.1f}%")
        print(f"    F1 Macro: {metrics['f1_macro']:.3f}")
        print(f"    Class 0 F1: {metrics['f1_class_0']:.3f}")
        print(f"    Class 1 F1: {metrics['f1_class_1']:.3f}")
        print(f"    Class 2 F1: {metrics['f1_class_2']:.3f}")
        print(f"    Class 3 F1: {metrics['f1_class_3']:.3f}")
    
    # Test 2: Calculate trading metrics
    print("\n💰 Test 2: Testing trading metrics...")
    
    trading_scenarios = {
        'Profitable Trader': [{'pnl': np.random.normal(50, 20)} for _ in range(100)],
        'Losing Trader': [{'pnl': np.random.normal(-30, 15)} for _ in range(100)],
        'Mixed Trader': [{'pnl': np.random.normal(0, 40)} for _ in range(100)],
        'High Win Rate, Small Wins': [
            {'pnl': 10 if np.random.random() < 0.8 else -30} for _ in range(100)
        ],
        'Low Win Rate, Large Wins': [
            {'pnl': 100 if np.random.random() < 0.3 else -20} for _ in range(100)
        ]
    }
    
    for name, trades in trading_scenarios.items():
        metrics = calculate_trading_metrics(trades)
        print(f"\n  {name}:")
        print(f"    Total Trades: {metrics['total_trades']}")
        print(f"    Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"    Total PnL: ${metrics['total_pnl']:.2f}")
        print(f"    Avg Win: ${metrics['avg_win']:.2f}")
        print(f"    Avg Loss: ${metrics['avg_loss']:.2f}")
        print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
        if 'sharpe_ratio' in metrics:
            print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Test 3: Test confusion matrix visualization
    print("\n📊 Test 3: Confusion matrix analysis...")
    
    y_pred_error = y_true.copy()
    # Introduce some errors
    error_indices = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
    y_pred_error[error_indices] = np.random.choice([0, 1, 2, 3], len(error_indices))
    
    metrics = calculate_imitation_metrics(y_true, y_pred_error)
    cm = np.array(metrics['confusion_matrix'])
    
    print(f"  Confusion Matrix:")
    print("      Pred 0  Pred 1  Pred 2  Pred 3")
    for i, row in enumerate(cm):
        print(f"  True {i}: {row}")
    
    # Calculate confusion metrics
    print(f"\n  Confusion Analysis:")
    for i in range(4):
        true_pos = cm[i][i]
        false_pos = np.sum(cm[:, i]) - true_pos
        false_neg = np.sum(cm[i, :]) - true_pos
        print(f"  Class {i}: TP={true_pos}, FP={false_pos}, FN={false_neg}")
    
    # Test 4: Edge cases
    print("\n⚠️ Test 4: Testing edge cases...")
    
    edge_cases = [
        ('Empty trades', []),
        ('Single winning trade', [{'pnl': 100}]),
        ('Single losing trade', [{'pnl': -50}]),
        ('All zero PnL', [{'pnl': 0} for _ in range(10)]),
        ('Very large numbers', [{'pnl': 1e6}, {'pnl': -5e5}]),
    ]
    
    for name, trades in edge_cases:
        metrics = calculate_trading_metrics(trades)
        print(f"  {name}: {len(metrics)} metrics computed")
    
    # Test 5: Performance comparison
    print("\n🏆 Test 5: Comparing strategy performance...")
    
    strategies = {
        'Buy & Hold': [{'pnl': 1000}],
        'Mean Reversion': [{'pnl': 50} for _ in range(20)],
        'Trend Following': [{'pnl': 200}, {'pnl': -50}, {'pnl': 300}, {'pnl': -30}],
        'Scalping': [{'pnl': 5} for _ in range(200)],
    }
    
    results = {}
    for name, trades in strategies.items():
        metrics = calculate_trading_metrics(trades)
        results[name] = {
            'total_pnl': metrics.get('total_pnl', 0),
            'win_rate': metrics.get('win_rate', 0),
            'sharpe': metrics.get('sharpe_ratio', 0)
        }
    
    # Find best strategy by Sharpe ratio
    best_strategy = max(results.keys(), key=lambda x: results[x]['sharpe'])
    print(f"\n  Best Strategy by Sharpe: {best_strategy}")
    for name, metrics in results.items():
        print(f"    {name:15} - PnL: ${metrics['total_pnl']:.0f}, "
              f"WR: {metrics['win_rate']*100:.0f}%, Sharpe: {metrics['sharpe']:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Metrics tests completed!")
    print("=" * 60)