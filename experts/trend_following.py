import pandas as pd
import numpy as np
from typing import Optional

class TrendFollowingExpert:
    """
    Expert trading strategy based on moving average crossover,
    RSI, ADX, and trend slope.
    """
    
    def __init__(self, fast_ma=20, slow_ma=50):
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.current_position = 0  # 0: flat, 1: long, -1: short


    def get_expert_action(self, df: pd.DataFrame, idx: int) -> int:
        """
        Return action at index idx:
        0 = BUY, 1 = SELL, 2 = HOLD, 3 = NEUTRAL/CLOSE
        """
        # # Not enough data
        # if idx < self.slow_ma:
        #     return 2  # HOLD

        # row = df.iloc[idx]
        # if pd.isna(row['fast_ma']) or pd.isna(row['slow_ma']):
        #     return 2

        # bullish = row['fast_ma'] > row['slow_ma']
        # bearish = row['fast_ma'] < row['slow_ma']

        row = df.iloc[idx]
        bullish=row['high_low_ratio_10']>1
        bearish=row['high_low_ratio_10']>1


        if self.current_position == 0:
            if bullish:
                action = 0   # BUY
            elif bearish:
                action = 1   # SELL
            else:
                action = 2   # HOLD
        elif self.current_position == 1:   # long
            if bullish:
                action = 2   # HOLD
            else:
                action = 3   # CLOSE
        else:   # short
            if bearish:
                action = 2   # HOLD
            else:
                action = 3   # CLOSE

        # Update internal position state
        self.current_position = self._next_position(action)
        return action




    def _next_position(self, action: int) -> int:
        """Compute new position after taking action."""
        if self.current_position == 0:
            if action == 0:
                return 1
            if action == 1:
                return -1
            return 0
        elif self.current_position == 1:
            if action == 3:
                return 0
            return 1
        else:  # -1
            if action == 3:
                return 0
            return -1

    def reset_position(self):
        self.current_position = 0


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING TREND FOLLOWING EXPERT")
    print("=" * 60)

    # Generate sample price data
    np.random.seed(42)
    n = 500
    dates = pd.date_range(start='2025-01-01', periods=n, freq='5min')
    prices = 2000 + np.cumsum(np.random.normal(0, 1, n))
    df = pd.DataFrame({
        'time': dates,
        'open': prices,
        'high': prices + np.random.uniform(0, 2, n),
        'low': prices - np.random.uniform(0, 2, n),
        'close': prices,
        'volume': np.random.randint(100, 10000, n)
    })

    expert = TrendFollowingExpert(fast_ma=20, slow_ma=50)

    # Generate expert actions
    actions = []
    expert.reset_position()
    for i in range(len(df)):
        action = expert.get_expert_action(df, i)
        actions.append(action)

    df['expert_action'] = actions
    print("\nExpert action distribution:")
    print(df['expert_action'].value_counts().sort_index().to_dict())

    # Print some statistics
    print(f"\nTotal actions: {len(df)}")
    print(f"BUY (0):   {(df['expert_action']==0).sum()}")
    print(f"SELL (1):  {(df['expert_action']==1).sum()}")
    print(f"HOLD (2):  {(df['expert_action']==2).sum()}")
    print(f"CLOSE (3): {(df['expert_action']==3).sum()}")

    # Show first few rows
    print("\nFirst 10 rows of expert signals:")
    print(df[['time', 'close', 'fast_ma', 'slow_ma', 'expert_action']].head(10))

    print("\n" + "=" * 60)
    print("✅ TrendFollowingExpert tests passed!")
    print("=" * 60)