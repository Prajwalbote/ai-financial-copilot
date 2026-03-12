"""
ml_models/feature_engineering.py — Technical Indicators
=========================================================
WHY FEATURE ENGINEERING?
    Raw OHLCV data (Open, High, Low, Close, Volume) alone
    is weak for prediction. Technical indicators transform
    raw prices into meaningful signals that capture:
    - Momentum (is price accelerating?)
    - Trend (is price going up or down?)
    - Volatility (how much is price swinging?)
    - Volume patterns (is movement backed by volume?)

WHAT YOU LEARN:
    - Feature engineering for time series
    - Common financial technical indicators
    - How to use pandas for financial calculations
    - Why feature quality matters more than model complexity
"""

import pandas as pd
import numpy as np
from utils.logger import get_logger

log = get_logger(__name__)


class FeatureEngineer:
    """
    Creates technical indicator features from stock price data.

    Input:  Raw OHLCV dataframe from yfinance
    Output: Dataframe with 20+ technical features

    Usage:
        fe = FeatureEngineer()
        features = fe.create_features(df)
    """

    def __init__(self):
        log.info("FeatureEngineer initialized")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all technical indicator features.

        Args:
            df: DataFrame with columns:
                Open, High, Low, Close, Volume

        Returns:
            DataFrame with original + feature columns
        """
        df = df.copy()

        # Ensure column names are clean
        df.columns = [c.lower() for c in df.columns]

        log.info(f"Creating features for {len(df)} data points")

        # Add each indicator group
        df = self._add_moving_averages(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df)
        df = self._add_volume_features(df)
        df = self._add_price_features(df)
        df = self._add_target(df)

        # Drop rows with NaN (from rolling calculations)
        df = df.dropna()

        log.info(
            f"Features created | "
            f"shape={df.shape} | "
            f"features={list(df.columns)}"
        )

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simple and Exponential Moving Averages.

        WHY MOVING AVERAGES?
        They smooth out daily noise to reveal the underlying trend.
        When short MA crosses above long MA = bullish signal.
        When short MA crosses below long MA = bearish signal.
        """
        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # Exponential Moving Average — gives more weight to recent prices
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

        # Price relative to moving averages (normalized)
        df['price_to_sma20'] = df['close'] / df['sma_20']
        df['price_to_sma50'] = df['close'] / df['sma_50']

        return df

    def _add_rsi(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """
        Relative Strength Index (RSI).

        RSI measures speed and magnitude of price changes.
        Range: 0 to 100

        > 70: Overbought — price may be due for correction
        < 30: Oversold  — price may be due for bounce
        50:   Neutral

        WHY RSI?
        One of the most reliable momentum indicators.
        Used by virtually every professional trader.
        """
        delta = df['close'].diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Average gains and losses over period
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()

        # Relative Strength
        rs = avg_gain / avg_loss

        # RSI formula
        df['rsi'] = 100 - (100 / (1 + rs))

        # RSI signal zones
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)

        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence).

        MACD = EMA(12) - EMA(26)
        Signal = EMA(9) of MACD
        Histogram = MACD - Signal

        Bullish signal: MACD crosses above signal line
        Bearish signal: MACD crosses below signal line

        WHY MACD?
        Captures both trend direction AND momentum.
        The histogram shows the strength of the signal.
        """
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # MACD crossover signal
        df['macd_crossover'] = (
            (df['macd'] > df['macd_signal']) &
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        ).astype(int)

        return df

    def _add_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Bollinger Bands.

        Middle Band = 20-day SMA
        Upper Band  = Middle + 2 * std deviation
        Lower Band  = Middle - 2 * std deviation

        When price touches upper band = potentially overbought
        When price touches lower band = potentially oversold
        Bands narrowing = low volatility (breakout coming?)

        WHY BOLLINGER BANDS?
        Combines trend AND volatility in one indicator.
        """
        rolling_mean = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()

        df['bb_upper'] = rolling_mean + (std_dev * rolling_std)
        df['bb_lower'] = rolling_mean - (std_dev * rolling_std)
        df['bb_middle'] = rolling_mean

        # Bandwidth — measures volatility
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Position within bands (0 = lower, 1 = upper)
        df['bb_position'] = (
            (df['close'] - df['bb_lower']) /
            (df['bb_upper'] - df['bb_lower'])
        )

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume-based features.

        WHY VOLUME?
        Price moves on high volume are more significant.
        A price rise on low volume may not be sustainable.
        Volume confirms or contradicts price movements.
        """
        df['volume_sma'] = df['volume'].rolling(window=20).mean()

        # Volume relative to average
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # On-Balance Volume (OBV)
        # Cumulative volume that follows price direction
        obv = []
        obv_val = 0
        for i in range(len(df)):
            if i == 0:
                obv.append(0)
                continue
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv_val += df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv_val -= df['volume'].iloc[i]
            obv.append(obv_val)

        df['obv'] = obv
        df['obv_sma'] = pd.Series(obv).rolling(window=10).mean().values

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Price-derived features.

        Daily returns, volatility, and momentum.
        """
        # Daily return
        df['daily_return'] = df['close'].pct_change()

        # Volatility (rolling std of returns)
        df['volatility'] = df['daily_return'].rolling(window=20).std()

        # Momentum (price change over N days)
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

        # High-Low range as % of close
        df['hl_range'] = (df['high'] - df['low']) / df['close']

        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create prediction target.

        Target: Will price be HIGHER tomorrow than today?
            1 = price goes up tomorrow
            0 = price goes down or stays same tomorrow

        WHY NEXT DAY?
        Short-term prediction is more reliable than
        long-term. Most algorithmic trading uses 1-5 day horizons.
        """
        df['target'] = (
            df['close'].shift(-1) > df['close']
        ).astype(int)

        return df

    def get_feature_columns(self) -> list[str]:
        """Return list of feature column names for model training."""
        return [
            'sma_5', 'sma_20', 'sma_50',
            'ema_12', 'ema_26',
            'price_to_sma20', 'price_to_sma50',
            'rsi', 'rsi_overbought', 'rsi_oversold',
            'macd', 'macd_signal', 'macd_histogram', 'macd_crossover',
            'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'volume_ratio', 'obv',
            'daily_return', 'volatility',
            'momentum_5', 'momentum_20',
            'hl_range'
        ]