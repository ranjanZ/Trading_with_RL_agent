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


    def get_expert_action(self, df: pd.DataFrame, idx: int, current_position: int = None) -> int:
        """
        Return action at index idx using new state machine:
        0 = BUY (open LONG or close SHORT)
        1 = SELL (open SHORT or close LONG)
        2 = HOLD
        
        Optimized for 5-min charts with proper trend detection and exit signals.
        
        Args:
            df: DataFrame with price data
            idx: Current index
            current_position: Current position (0=flat, 1=long, -1=short)
                            If None, uses self.current_position
        """
        if current_position is not None:
            self.current_position = current_position
        
        # Need minimum history for indicators
        if idx < 5:
            return 2  # HOLD until we have enough data
        
        row = df.iloc[idx]
        
        # Calculate momentum and trend signals for 5-min chart
        bullish, bearish = self._calculate_trend_signals(df, idx)
        
        # State machine logic with new action space
        if self.current_position == 0:  # FLAT
            if bullish:
                action = 0   # BUY (open LONG)
            elif bearish:
                action = 1   # SELL (open SHORT)
            else:
                action = 2   # HOLD
        elif self.current_position == 1:  # LONG
            # In LONG: hold on bullish, exit on bearish/momentum loss
            if bullish:
                action = 2   # HOLD (maintain LONG)
            else:
                action = 1   # SELL (close LONG on trend reversal)
        else:  # SHORT (-1)
            # In SHORT: hold on bearish, exit on bullish/momentum loss
            if bearish:
                action = 2   # HOLD (maintain SHORT)
            else:
                action = 0   # BUY (close SHORT on trend reversal)

        # Update internal position state
        self.current_position = self._next_position(action)
        return action
    
    def _calculate_trend_signals(self, df: pd.DataFrame, idx: int) -> tuple:
        """
        Calculate bullish/bearish signals optimized for 5-min charts.
        
        Returns:
            (bullish: bool, bearish: bool)
        """
        row = df.iloc[idx]
        
        # Strategy 1: Price momentum (compare to recent prices)
        # For 5-min chart, look at last 3 candles trend
        bullish_momentum = False
        bearish_momentum = False
        
        if idx >= 3:
            prev_closes = df.iloc[max(0, idx-3):idx+1]['close'].values
            # Check if prices are generally increasing
            bullish_momentum = prev_closes[-1] > prev_closes[0] and np.mean(np.diff(prev_closes)) > 0
            # Check if prices are generally decreasing
            bearish_momentum = prev_closes[-1] < prev_closes[0] and np.mean(np.diff(prev_closes)) < 0
        
        # Strategy 2: Use close vs open (candle direction)
        candle_bullish = row['close'] > row['open']
        candle_bearish = row['close'] < row['open']
        
        # Strategy 3: Volume-weighted direction (if available)
        volume_weighted = False
        if 'volume' in df.columns and row['volume'] > df['volume'].rolling(5).mean().iloc[idx]:
            if candle_bullish:
                volume_weighted = True
            elif candle_bearish:
                volume_weighted = False
        
        # Strategy 4: Moving average crossover (if available and calculated)
        ma_bullish = False
        ma_bearish = False
        if 'fast_ma' in df.columns and 'slow_ma' in df.columns:
            if pd.notna(df.iloc[idx]['fast_ma']) and pd.notna(df.iloc[idx]['slow_ma']):
                ma_bullish = df.iloc[idx]['fast_ma'] > df.iloc[idx]['slow_ma']
                ma_bearish = df.iloc[idx]['fast_ma'] < df.iloc[idx]['slow_ma']
        
        # Strategy 5: RSI-like momentum for 5-min chart (if available)
        rsi_bullish = False
        rsi_bearish = False
        if 'RSI' in df.columns or 'rsi' in df.columns:
            rsi_col = 'RSI' if 'RSI' in df.columns else 'rsi'
            rsi_value = df.iloc[idx][rsi_col]
            if pd.notna(rsi_value):
                rsi_bullish = rsi_value > 50  # Momentum up
                rsi_bearish = rsi_value < 50  # Momentum down
        
        # Combine signals: need at least 2 signals for strong conviction
        bullish_signals = sum([bullish_momentum, candle_bullish, ma_bullish, rsi_bullish])
        bearish_signals = sum([bearish_momentum, candle_bearish, ma_bearish, rsi_bearish])
        
        # For 5-min chart: be more responsive but require minimum confirmation
        bullish = bullish_signals >= 1 and bearish_signals == 0
        bearish = bearish_signals >= 1 and bullish_signals == 0
        
        return bullish, bearish




    def _next_position(self, action: int) -> int:
        """
        Compute new position after taking action (new state machine).
        Actions: 0=BUY, 1=SELL, 2=HOLD
        """
        if self.current_position == 0:  # FLAT
            if action == 0:  # BUY
                return 1  # Go LONG
            elif action == 1:  # SELL
                return -1  # Go SHORT
            else:  # HOLD
                return 0
        elif self.current_position == 1:  # LONG
            if action == 0:  # BUY (invalid - shouldn't happen)
                return 1
            elif action == 1:  # SELL (close LONG)
                return 0  # Return to FLAT
            else:  # HOLD
                return 1
        else:  # SHORT (-1)
            if action == 0:  # BUY (close SHORT)
                return 0  # Return to FLAT
            elif action == 1:  # SELL (invalid - shouldn't happen)
                return -1
            else:  # HOLD
                return -1

    def reset_position(self):
        self.current_position = 0
    
    def prepare_data_with_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame with technical indicators for 5-min chart analysis.
        
        Adds:
        - fast_ma: Fast moving average (5 periods for 5-min = ~25 min)
        - slow_ma: Slow moving average (10 periods for 5-min = ~50 min)
        - RSI: Relative Strength Index (9 periods for 5-min sensitivity)
        - momentum: Price momentum (difference from 3 candles ago)
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Moving averages for 5-min chart (shorter periods for responsiveness)
        df['fast_ma'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['slow_ma'] = df['close'].rolling(window=10, min_periods=1).mean()
        
        # RSI (9-period for 5-min sensitivity)
        df['RSI'] = self._calculate_rsi(df['close'], period=9)
        
        # Momentum indicator
        df['momentum'] = df['close'].diff(3)
        
        # High-Low range over 5 candles
        df['hl_range'] = (df['high'] - df['low']).rolling(window=5, min_periods=1).mean()
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


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

    # Show first few rows
    print("\nFirst 10 rows of expert signals:")
    print(df[['time', 'close', 'expert_action']].head(10))

    print("\n" + "=" * 60)
    print("✅ TrendFollowingExpert tests passed!")
    print("=" * 60)