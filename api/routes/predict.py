"""
api/routes/predict.py — Stock Prediction Endpoints
"""

from fastapi import APIRouter, HTTPException
from api.schemas import PredictionResponse, RiskResponse
from ml_models.stock_predictor import StockPredictor
from ml_models.risk_analyzer import RiskAnalyzer
from utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["Stock Prediction"])

# Shared instances — loaded once, reused per request
_predictor = None
_risk_analyzer = None


def get_predictor() -> StockPredictor:
    global _predictor
    if _predictor is None:
        _predictor = StockPredictor()
    return _predictor


def get_risk_analyzer() -> RiskAnalyzer:
    global _risk_analyzer
    if _risk_analyzer is None:
        _risk_analyzer = RiskAnalyzer()
    return _risk_analyzer


@router.get("/predict/{ticker}", response_model=PredictionResponse)
async def predict_stock(ticker: str):
    """
    Predict next-day price direction for a stock.

    - ticker: Stock symbol (e.g. AAPL, MSFT, TSLA)
    - Returns direction (UP/DOWN) with confidence score
    - Returns current price and key technical indicators
    """
    ticker = ticker.upper()
    log.info(f"Prediction request for: {ticker}")

    try:
        predictor = get_predictor()

        # Train if no model exists for this ticker
        import os
        model_path = f"data/models/{ticker}_predictor.pkl"
        if not os.path.exists(model_path):
            log.info(f"No model found for {ticker}, training now...")
            metrics = predictor.train(ticker)
            if "error" in metrics:
                raise HTTPException(
                    status_code=400,
                    detail=metrics["error"]
                )

        result = predictor.predict(ticker)

        if "error" in result:
            raise HTTPException(
                status_code=400,
                detail=result["error"]
            )

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error predicting {ticker}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error predicting stock: {str(e)}"
        )


@router.get("/risk/{ticker}", response_model=RiskResponse)
async def analyze_risk(ticker: str):
    """
    Analyze financial risk metrics for a stock.

    - Returns risk score (0-100)
    - Returns volatility, VaR, Sharpe ratio, max drawdown
    - Returns human readable interpretation
    """
    ticker = ticker.upper()
    log.info(f"Risk analysis request for: {ticker}")

    try:
        analyzer = get_risk_analyzer()
        result = analyzer.analyze(ticker)

        if "error" in result:
            raise HTTPException(
                status_code=400,
                detail=result["error"]
            )

        return RiskResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error analyzing risk for {ticker}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing risk: {str(e)}"
        )