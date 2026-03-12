"""
ml_models/risk_analyzer.py — Financial Risk Scoring
====================================================
WHY RISK ANALYSIS?
    Knowing price direction is only half the picture.
    Risk analysis tells you HOW RISKY the position is.

    A stock can be predicted to go UP but with HIGH risk.
    A smart investor wants high confidence + low risk.

WHAT YOU LEARN:
    - Risk metrics (VaR, Sharpe, volatility)
    - How to combine multiple signals into a score
    - Portfolio risk concepts
"""

import numpy as np
import pandas as pd
import yfinance as yf
from utils.logger import get_logger

log = get_logger(__name__)


class RiskAnalyzer:
    """
    Calculates financial risk metrics for stocks.

    Metrics calculated:
        - Volatility (how much price swings)
        - Value at Risk (worst expected loss)
        - Sharpe Ratio (return per unit of risk)
        - Max Drawdown (biggest historical drop)
        - Risk Score (0-100 composite score)

    Usage:
        analyzer = RiskAnalyzer()
        risk = analyzer.analyze("AAPL")
        print(risk['risk_score'])    # 0-100
        print(risk['risk_level'])    # LOW/MEDIUM/HIGH
    """

    def __init__(self):
        log.info("RiskAnalyzer initialized")

    def analyze(self, ticker: str, period: str = "1y") -> dict:
        """
        Full risk analysis for a stock.

        Args:
            ticker: Stock symbol
            period: Historical period for analysis

        Returns:
            Dict with all risk metrics and overall score
        """
        log.info(f"Analyzing risk for {ticker}")

        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            return {"error": f"No data for {ticker}"}

        returns = df['Close'].pct_change().dropna()

        # Calculate individual metrics
        volatility = self._calculate_volatility(returns)
        var_95 = self._calculate_var(returns)
        sharpe = self._calculate_sharpe(returns)
        max_dd = self._calculate_max_drawdown(df['Close'])
        risk_score = self._calculate_risk_score(
            volatility, var_95, sharpe, max_dd
        )

        result = {
            "ticker": ticker,
            "risk_score": round(risk_score, 1),
            "risk_level": self._get_risk_level(risk_score),
            "metrics": {
                "annual_volatility": round(volatility * 100, 2),
                "value_at_risk_95": round(var_95 * 100, 2),
                "sharpe_ratio": round(sharpe, 3),
                "max_drawdown": round(max_dd * 100, 2),
            },
            "interpretation": self._interpret_risk(
                risk_score, volatility, var_95, sharpe
            )
        }

        log.info(
            f"Risk analysis complete | "
            f"{ticker} | "
            f"score={risk_score:.1f} | "
            f"level={result['risk_level']}"
        )

        return result

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """
        Annual volatility = std dev of daily returns * sqrt(252)

        WHY 252?
        There are ~252 trading days in a year.
        Multiplying by sqrt(252) annualizes the daily volatility.
        """
        return returns.std() * np.sqrt(252)

    def _calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Value at Risk (VaR) at 95% confidence.

        Interpretation:
        VaR = -0.03 means:
        "With 95% confidence, we won't lose more than 3% in a day"

        WHY VAR?
        Standard risk metric used by every bank and hedge fund.
        Required by financial regulators.
        """
        return np.percentile(returns, (1 - confidence) * 100)

    def _calculate_sharpe(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Sharpe Ratio = (Return - Risk Free Rate) / Volatility

        Interpretation:
        > 1.0 = Good risk-adjusted return
        > 2.0 = Very good
        < 0   = Worse than risk-free (just buy T-bills!)

        WHY SHARPE?
        Measures return PER UNIT OF RISK taken.
        A stock returning 20% with huge volatility may be
        worse than a stock returning 12% with low volatility.
        """
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)

        if annual_vol == 0:
            return 0

        return (annual_return - risk_free_rate) / annual_vol

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        Maximum Drawdown = biggest peak-to-trough decline.

        If stock went from $100 → $150 → $90:
        Max drawdown = (90 - 150) / 150 = -40%

        WHY MAX DRAWDOWN?
        Shows the worst case historical loss.
        Critical for understanding downside risk.
        """
        rolling_max = prices.expanding().max()
        drawdowns = (prices - rolling_max) / rolling_max
        return float(drawdowns.min())

    def _calculate_risk_score(
        self,
        volatility: float,
        var: float,
        sharpe: float,
        max_drawdown: float
    ) -> float:
        """
        Composite risk score from 0 to 100.
        Higher = more risky.

        Weights:
            Volatility:    30%
            VaR:           30%
            Sharpe:        20% (inverted — higher sharpe = lower risk)
            Max Drawdown:  20%
        """
        # Normalize each metric to 0-100 scale
        vol_score = min(volatility * 200, 100)
        var_score = min(abs(var) * 1000, 100)
        sharpe_score = max(0, min(100 - (sharpe * 20), 100))
        dd_score = min(abs(max_drawdown) * 200, 100)

        # Weighted composite
        risk_score = (
            vol_score * 0.30 +
            var_score * 0.30 +
            sharpe_score * 0.20 +
            dd_score * 0.20
        )

        return risk_score

    def _get_risk_level(self, score: float) -> str:
        """Convert numeric score to risk level label."""
        if score < 30:
            return "LOW"
        elif score < 60:
            return "MEDIUM"
        else:
            return "HIGH"

    def _interpret_risk(
        self,
        score: float,
        volatility: float,
        var: float,
        sharpe: float
    ) -> str:
        """Generate human readable risk interpretation."""
        level = self._get_risk_level(score)

        return (
            f"Risk Level: {level} (score: {score:.1f}/100). "
            f"Annual volatility of {volatility*100:.1f}%. "
            f"At 95% confidence, maximum daily loss is {abs(var)*100:.1f}%. "
            f"Sharpe ratio of {sharpe:.2f} indicates "
            f"{'good' if sharpe > 1 else 'poor'} risk-adjusted returns."
        )