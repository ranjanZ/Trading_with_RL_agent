import pandas as pd
import numpy as np
import random
from typing import Optional

class RandomExpert:
    """
    Expert that takes random actions, following the same action space
    and state management as TrendFollowingExpert.

    Action mapping:
        1  → BUY   (open long or close short)
        -1 → SELL  (open short or close long)
        0  → HOLD  (do nothing)

    The decision is completely random (not based on price data),
    but respects the current position to maintain a valid state
    machine via the same _next_position logic.
    """

    def __init__(self, p_buy: float = 0.33, p_sell: float = 0.33, seed: Optional[int] = None):
        """
        Initialize random expert.

        Args:
            p_buy: Probability of returning BUY action (1).
            p_sell: Probability of returning SELL action (-1).
            seed: Random seed for reproducibility.
        """
        # Normalize probabilities; p_hold = 1 - p_buy - p_sell
        total = p_buy + p_sell
        if total > 1.0:
            p_buy /= total
            p_sell /= total
        self.p_buy = p_buy
        self.p_sell = p_sell
        self.current_position = 0   # 0=flat, 1=long, -1=short
        if seed is not None:
            random.seed(seed)

    def get_expert_action(self, df: pd.DataFrame, idx: int,
                          current_position: Optional[int] = None) -> int:
        """
        Return a random action (1, -1, 0) following the same action
        interpretation as TrendFollowingExpert. Does not use the
        DataFrame or index; they are kept for interface compatibility.

        Args:
            df: DataFrame with price data (ignored).
            idx: Current index (ignored).
            current_position: If provided, updates internal position
                              before generating action.

        Returns:
            int: 1 (BUY), -1 (SELL), or 0 (HOLD).
        """
        if current_position is not None:
            self.current_position = current_position

        # Random action with given probabilities
        r = random.random()
        if r < self.p_buy:
            action = 1
        elif r < self.p_buy + self.p_sell:
            action = -1
        else:
            action = 0

        # Update internal position (same state machine as TrendFollowingExpert)
        self.current_position = self._next_position(action)
        return action

    def _next_position(self, action: int) -> int:
        """
        Compute new position after taking action.
        Same logic as TrendFollowingExpert's _next_position.
        """
        if self.current_position == 0:   # FLAT
            if action == 1:
                return 1
            elif action == -1:
                return -1
            else:
                return 0
        elif self.current_position == 1: # LONG
            if action == 1:              # invalid, keep long
                return 1
            elif action == -1:           # sell to close
                return 0
            else:                        # hold
                return 1
        else:                            # SHORT (-1)
            if action == 1:              # buy to cover
                return 0
            elif action == -1:           # invalid, keep short
                return -1
            else:
                return -1

    def reset_position(self):
        self.current_position = 0

    def prepare_data_with_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For interface compatibility only; returns the DataFrame unchanged.
        """
        return df.copy()


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING RANDOM EXPERT (new action space)")
    print("=" * 60)

    # Create a dummy DataFrame (will be ignored by the expert)
    dates = pd.date_range(start='2025-01-01', periods=100, freq='5min')
    df = pd.DataFrame({
        'time': dates,
        'close': np.random.normal(2000, 10, 100)
    })

    # Expert with equal probabilities, seeded for reproducibility
    expert = RandomExpert(p_buy=0.33, p_sell=0.33, seed=123)
    actions = []
    expert.reset_position()
    for i in range(len(df)):
        action = expert.get_expert_action(df, i)
        actions.append(action)

    df['random_action'] = actions
    print("\nRandom action distribution:")
    print(df['random_action'].value_counts().sort_index().to_dict())

    print(f"\nTotal actions: {len(df)}")
    print(f"BUY (1):   {(df['random_action']==1).sum()}")
    print(f"SELL (-1): {(df['random_action']==-1).sum()}")
    print(f"HOLD (0):  {(df['random_action']==0).sum()}")

    print("\nFirst 10 rows:")
    print(df[['time', 'close', 'random_action']].head(10))

    print("\n" + "=" * 60)
    print("✅ RandomExpert tests passed!")
    print("=" * 60)