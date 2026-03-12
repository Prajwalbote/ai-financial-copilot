"""
ml_models/stock_predictor.py — Improved Stock Predictor
========================================================
IMPROVEMENTS:
    1. XGBoost added alongside Random Forest
    2. Model comparison — picks the best one
    3. 5 years of training data instead of 2
    4. Multiple stock training support
    5. Better evaluation metrics
    6. Fixed UndefinedMetricWarning

WHY XGBOOST OVER RANDOM FOREST?
    Random Forest:  parallel trees, majority vote
    XGBoost:        sequential trees, error correction
    
    XGBoost almost always wins on tabular financial data
    because it learns from its own mistakes each iteration.
    Used by winning teams in almost every Kaggle competition.
"""

import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from xgboost import XGBClassifier
from ml_models.feature_engineering import FeatureEngineer
from utils.logger import get_logger
from utils.config import get_config

log = get_logger(__name__)


class StockPredictor:
    """
    Predicts next-day stock price direction.

    Uses both Random Forest AND XGBoost.
    Automatically picks the better performing model.

    Usage:
        predictor = StockPredictor()
        predictor.train("AAPL")
        result = predictor.predict("AAPL")
        print(result['direction'])    # "UP" or "DOWN"
        print(result['confidence'])   # 0.0 to 1.0
        print(result['model_used'])   # which model won
    """

    def __init__(self):
        self.cfg = get_config()
        self.fe = FeatureEngineer()
        self.model = None
        self.model_type = None
        self.feature_cols = self.fe.get_feature_columns()
        self.model_dir = Path("data/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        log.info("StockPredictor initialized")

    def fetch_data(
        self,
        ticker: str,
        period: str = "5y"      # IMPROVED: 5 years instead of 2
    ) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.

        WHY 5 YEARS?
        More data = model sees more market cycles.
        2 years might only cover a bull market.
        5 years includes corrections, crashes, recoveries.
        Better generalization = better accuracy.
        """
        log.info(f"Fetching data for {ticker} | period={period}")

        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            log.error(f"No data found for ticker: {ticker}")
            return pd.DataFrame()

        log.info(f"Fetched {len(df)} days of data for {ticker}")
        return df

    def _build_random_forest(self) -> RandomForestClassifier:
        """Build Random Forest model with optimized params."""
        return RandomForestClassifier(
            n_estimators=200,        # More trees = more stable
            max_depth=8,             # Slightly shallower = less overfit
            min_samples_split=20,
            min_samples_leaf=10,     # NEW: minimum samples per leaf
            max_features='sqrt',     # NEW: random feature subset per split
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # NEW: handle imbalanced UP/DOWN
        )

    def _build_xgboost(self) -> XGBClassifier:
        """
        Build XGBoost model.

        KEY PARAMETERS:
        n_estimators:     number of boosting rounds
        max_depth:        tree depth (3-6 works well for finance)
        learning_rate:    how much each tree contributes
                          smaller = more trees needed but better
        subsample:        use 80% of data per tree (prevents overfit)
        colsample_bytree: use 80% of features per tree
        eval_metric:      what to optimize (logloss for binary)
        """
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=0             # Suppress XGBoost output
        )

    def train(self, ticker: str) -> dict:
        """
        Train both models and keep the better one.

        Full pipeline:
            Fetch 5y data → Engineer features →
            Time split → Train RF + XGB →
            Compare accuracy → Save best model
        """
        log.info(f"Training StockPredictor for {ticker}")

        # Step 1 — Fetch and engineer features
        raw_df = self.fetch_data(ticker)
        if raw_df.empty:
            return {"error": f"No data for {ticker}"}

        df = self.fe.create_features(raw_df)

        # Step 2 — Prepare features and target
        X = df[self.feature_cols].values
        y = df['target'].values

        # Step 3 — Time series split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        log.info(
            f"Data split | "
            f"train={len(X_train)} | "
            f"test={len(X_test)}"
        )

        # Step 4 — Train Random Forest
        log.info("Training Random Forest...")
        rf_model = self._build_random_forest()
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        log.info(f"Random Forest accuracy: {rf_accuracy:.4f}")

        # Step 5 — Train XGBoost
        log.info("Training XGBoost...")
        xgb_model = self._build_xgboost()
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        xgb_pred = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        log.info(f"XGBoost accuracy: {xgb_accuracy:.4f}")

        # Step 6 — Pick the winner
        if xgb_accuracy >= rf_accuracy:
            self.model = xgb_model
            self.model_type = "XGBoost"
            best_accuracy = xgb_accuracy
            best_pred = xgb_pred
        else:
            self.model = rf_model
            self.model_type = "RandomForest"
            best_accuracy = rf_accuracy
            best_pred = rf_pred

        log.info(
            f"Best model: {self.model_type} | "
            f"accuracy={best_accuracy:.4f}"
        )

        metrics = {
            "ticker": ticker,
            "best_model": self.model_type,
            "accuracy": round(best_accuracy, 4),
            "random_forest_accuracy": round(rf_accuracy, 4),
            "xgboost_accuracy": round(xgb_accuracy, 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "classification_report": classification_report(
                y_test,
                best_pred,
                target_names=["DOWN", "UP"],
                zero_division=0     # FIXED: no more warnings
            ),
            "feature_importance": self._get_feature_importance()
        }

        # Save best model
        self._save_model(ticker)

        return metrics

    def train_multiple(self, tickers: list[str]) -> dict:
        """
        Train on multiple stocks and combine results.

        WHY MULTI-STOCK TRAINING?
        A model trained on AAPL only learns Apple-specific
        patterns. Training on many stocks learns GENERAL
        market patterns that apply everywhere.

        This is called transfer learning for tabular data.
        """
        log.info(f"Multi-stock training | tickers={tickers}")

        all_X_train = []
        all_y_train = []
        results = {}

        for ticker in tickers:
            raw_df = self.fetch_data(ticker)
            if raw_df.empty:
                log.warning(f"Skipping {ticker} — no data")
                continue

            df = self.fe.create_features(raw_df)
            X = df[self.feature_cols].values
            y = df['target'].values

            # Use 80% for training from each stock
            split_idx = int(len(X) * 0.8)
            all_X_train.append(X[:split_idx])
            all_y_train.append(y[:split_idx])

            results[ticker] = len(X)
            log.info(f"Added {ticker}: {len(X)} samples")

        if not all_X_train:
            return {"error": "No data collected"}

        # Combine all stocks into one training set
        X_combined = np.vstack(all_X_train)
        y_combined = np.concatenate(all_y_train)

        log.info(
            f"Combined dataset | "
            f"samples={len(X_combined)} | "
            f"stocks={len(tickers)}"
        )

        # Train XGBoost on combined data
        self.model = self._build_xgboost()
        self.model_type = "XGBoost_MultiStock"
        self.model.fit(X_combined, y_combined, verbose=False)

        # Save as generic multi-stock model
        self._save_model("MULTI")

        return {
            "model": self.model_type,
            "total_samples": len(X_combined),
            "stocks_trained": results
        }

    def predict(self, ticker: str) -> dict:
        """
        Predict next day price direction for a ticker.

        Returns direction, confidence, price, and indicators.
        """
        if self.model is None:
            # Try loading ticker-specific model first
            if not self._load_model(ticker):
                # Fall back to multi-stock model
                if not self._load_model("MULTI"):
                    return {"error": "No model found. Run train() first."}

        raw_df = self.fetch_data(ticker, period="3mo")
        if raw_df.empty:
            return {"error": f"No data for {ticker}"}

        df = self.fe.create_features(raw_df)

        if len(df) == 0:
            return {"error": "Not enough data for prediction"}

        latest = df[self.feature_cols].iloc[-1].values.reshape(1, -1)
        latest_price = df['close'].iloc[-1]
        latest_rsi = df['rsi'].iloc[-1]
        latest_macd = df['macd'].iloc[-1]

        prediction = self.model.predict(latest)[0]
        probabilities = self.model.predict_proba(latest)[0]
        confidence = float(max(probabilities))
        direction = "UP" if prediction == 1 else "DOWN"

        result = {
            "ticker": ticker,
            "direction": direction,
            "confidence": round(confidence, 4),
            "current_price": round(float(latest_price), 2),
            "model_used": self.model_type or "unknown",
            "indicators": {
                "rsi": round(float(latest_rsi), 2),
                "macd": round(float(latest_macd), 4),
                "signal": "overbought" if latest_rsi > 70
                          else "oversold" if latest_rsi < 30
                          else "neutral"
            }
        }

        log.info(
            f"Prediction for {ticker} | "
            f"direction={direction} | "
            f"confidence={confidence:.4f} | "
            f"model={self.model_type}"
        )

        return result

    def _get_feature_importance(self) -> dict:
        """Get top 10 most important features."""
        if self.model is None:
            return {}

        importance = dict(zip(
            self.feature_cols,
            self.model.feature_importances_
        ))

        sorted_importance = dict(
            sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        )

        return {k: round(v, 4) for k, v in sorted_importance.items()}

    def _save_model(self, ticker: str):
        """Save trained model to disk."""
        model_file = self.model_dir / f"{ticker}_predictor.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump({
                "model": self.model,
                "model_type": self.model_type
            }, f)
        log.info(f"Model saved: {model_file}")

    def _load_model(self, ticker: str) -> bool:
        """Load saved model from disk."""
        model_file = self.model_dir / f"{ticker}_predictor.pkl"
        if not model_file.exists():
            log.warning(f"No saved model for {ticker}")
            return False

        with open(model_file, 'rb') as f:
            saved = pickle.load(f)

        # Handle both old format and new format
        if isinstance(saved, dict):
            self.model = saved["model"]
            self.model_type = saved.get("model_type", "unknown")
        else:
            self.model = saved
            self.model_type = "unknown"

        log.info(
            f"Model loaded: {model_file} | "
            f"type={self.model_type}"
        )
        return True