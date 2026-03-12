import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.stock_predictor import StockPredictor
from ml_models.risk_analyzer import RiskAnalyzer

print("="*60)
print("TEST 1 — Train Single Stock (RF vs XGBoost)")
print("="*60)
predictor = StockPredictor()
metrics = predictor.train("AAPL")
print(f"Best Model:      {metrics['best_model']}")
print(f"Best Accuracy:   {metrics['accuracy']}")
print(f"Random Forest:   {metrics['random_forest_accuracy']}")
print(f"XGBoost:         {metrics['xgboost_accuracy']}")
print(f"Train samples:   {metrics['train_samples']}")
print(f"Test samples:    {metrics['test_samples']}")
print("\nTop Features:")
for feat, importance in metrics['feature_importance'].items():
    print(f"  {feat}: {importance}")

print("\n" + "="*60)
print("TEST 2 — Multi Stock Training")
print("="*60)
multi_result = predictor.train_multiple(["AAPL", "MSFT", "GOOGL"])
print(f"Model:          {multi_result['model']}")
print(f"Total Samples:  {multi_result['total_samples']}")
print(f"Stocks Trained: {list(multi_result['stocks_trained'].keys())}")

print("\n" + "="*60)
print("TEST 3 — Predict with Best Model")
print("="*60)
result = predictor.predict("AAPL")
print(f"Ticker:      {result['ticker']}")
print(f"Direction:   {result['direction']}")
print(f"Confidence:  {result['confidence']}")
print(f"Price:       ${result['current_price']}")
print(f"Model Used:  {result['model_used']}")
print(f"RSI:         {result['indicators']['rsi']}")
print(f"Signal:      {result['indicators']['signal']}")

print("\n" + "="*60)
print("TEST 4 — Risk Analysis")
print("="*60)
analyzer = RiskAnalyzer()
risk = analyzer.analyze("AAPL")
print(f"Risk Score:  {risk['risk_score']}/100")
print(f"Risk Level:  {risk['risk_level']}")
print(f"Volatility:  {risk['metrics']['annual_volatility']}%")
print(f"Sharpe:      {risk['metrics']['sharpe_ratio']}")