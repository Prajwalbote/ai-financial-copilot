# 🤖 AI Financial Research Copilot

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A production-grade RAG system for financial research.
Ask questions about financial documents in plain English
and get cited answers backed by source documents.

---

## Features
- Answer questions from financial documents (PDF/URL)
- Summarize financial reports with source citations
- Predict stock price direction using ML (Random Forest vs XGBoost — best model auto-selected)
- Analyze financial risk with institutional metrics (VaR, Sharpe Ratio, Max Drawdown)
- Ingest new documents via UI or API
- Response caching for fast repeated queries

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Fast, accurate, runs on CPU |
| LLM | google/flan-t5-base | Free, open source, CPU-friendly |
| Vector DB | FAISS (IndexFlatIP) | Fast similarity search |
| ML Models | Random Forest + XGBoost | Auto-picks best performer |
| Backend | FastAPI | Async, auto-docs, Pydantic validation |
| Frontend | Streamlit | Professional UI in pure Python |
| Fine-tuning | LoRA (PEFT) | 97% fewer trainable parameters |
| Deployment | Docker + Docker Compose | Run anywhere consistently |
| Caching | In-memory TTL cache | Instant repeated queries |

---

## ML Model: Random Forest vs XGBoost

One of the key improvements in this project is the **automatic model comparison** between Random Forest and XGBoost:
```
Random Forest:
  Builds many decision trees independently
  Each tree votes → majority wins
  Works well on smaller datasets
  Good baseline model

XGBoost (Gradient Boosting):
  Builds trees sequentially
  Each tree learns from previous tree's mistakes
  Better on larger datasets
  Used by winning teams in most Kaggle competitions
```

### How It Works
```python
# Train both models
rf_accuracy  = train_and_evaluate(RandomForestClassifier())
xgb_accuracy = train_and_evaluate(XGBClassifier())

# Auto-select the winner
best_model = xgb if xgb_accuracy >= rf_accuracy else rf
```

### Results on AAPL (5 years of data)
| Model | Accuracy | Training Samples |
|-------|----------|-----------------|
| Random Forest | 50.41% | 964 days |
| XGBoost | 50.00% | 964 days |
| **XGBoost MultiStock** | **trained on 2,892 samples** | AAPL + MSFT + GOOGL |

### Why ~50% Accuracy Is Correct
> The Efficient Market Hypothesis states that all public information
> is already priced into stock prices. A model using only public
> technical indicators cannot reliably exceed 50-55% accuracy.
> Renaissance Technologies, the world's best quant fund, achieves ~55%.
> Our result is honest and realistic — not a sign of a broken model.

### Top Predictive Features
| Feature | Importance | What It Measures |
|---------|-----------|-----------------|
| momentum_20 | 7.09% | 20-day price momentum |
| daily_return | 6.03% | Today's price change |
| volume_ratio | 5.34% | Unusual volume activity |
| bb_width | 5.17% | Volatility (Bollinger Band width) |
| bb_position | 4.86% | Where price sits in the band |
| price_to_sma20 | 4.64% | Distance from 20-day average |

### Multi-Stock Training
Training on multiple stocks teaches the model **general market patterns**
rather than stock-specific quirks:
```
Single stock:  learns Apple-specific behavior only
Multi stock:   learns general market behavior
               AAPL (1,206 samples)
             + MSFT (1,206 samples)
             + GOOGL (1,206 samples)
             = 2,892 combined training samples
```

---

## Architecture
```
                   ┌─────────────────┐
                   │   Streamlit UI  │
                   │  localhost:8501 │
                   └────────┬────────┘
                            │ HTTP POST /api/ask
                   ┌────────▼────────┐
                   │    FastAPI      │
                   │  localhost:8000 │
                   └────────┬────────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
    ┌────────▼───┐  ┌───────▼──────┐  ┌───▼────────┐
    │   Cache    │  │ RAG Pipeline │  │ ML Models  │
    │ (TTL 1hr)  │  │              │  │ RF + XGB   │
    └────────────┘  └───────┬──────┘  └────────────┘
                            │
             ┌──────────────┼──────────────┐
             │                             │
    ┌────────▼───────┐          ┌──────────▼──────┐
    │  FAISS Search  │          │   Flan-T5 LLM   │
    │  (retrieve)    │          │   (generate)    │
    └────────┬───────┘          └─────────────────┘
             │
    ┌────────▼───────┐
    │ PDF Chunks     │
    │ (384-dim vecs) │
    └────────────────┘
```

---

## Project Structure
```
financial_copilot/
├── ingestion/          # Document loading and chunking
├── embeddings/         # Embedding generation (384-dim)
├── vectordb/           # FAISS vector store
├── retrieval/          # RAG pipeline
├── llm/                # LLM wrapper and prompt templates
├── finetuning/         # LoRA fine-tuning (0.71% params)
├── ml_models/          # Stock prediction (RF + XGBoost)
├── api/                # FastAPI backend
├── ui/                 # Streamlit frontend
├── utils/              # Config, logger, cache
└── docker/             # Deployment files
```

---

## Quick Start

### Local Development
```bash
# 1. Clone and setup
git clone https://github.com/Prajwalbote/ai-financial-copilot
cd ai-financial-copilot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Start UI (new terminal)
streamlit run ui/app.py
```

### Docker Deployment
```bash
cd docker
docker-compose up --build

# UI:  http://localhost:8501
# API: http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/ask | Ask a financial question |
| POST | /api/summarize | Summarize documents |
| POST | /api/analyze-risks | Risk analysis from docs |
| GET | /api/predict/{ticker} | Stock direction prediction |
| GET | /api/risk/{ticker} | Risk metrics (VaR, Sharpe) |
| POST | /api/ingest/pdf | Upload and index a PDF |
| POST | /api/ingest/url | Ingest from URL |
| GET | /api/cache-stats | Cache hit rate and stats |
| GET | /health | System health check |

---

## Key Concepts Implemented

**RAG (Retrieval Augmented Generation)**
Retrieves relevant document chunks at query time and includes
them in the LLM prompt — solving the knowledge cutoff problem
and reducing hallucination.

**LoRA Fine-Tuning**
Fine-tuned Flan-T5 on financial QA pairs training only 0.71%
of parameters by adding low-rank decomposition matrices while
freezing original weights. Runs on CPU in minutes.

**Semantic Search**
Text converted to 384-dimensional vectors using
sentence-transformers. Cosine similarity finds relevant chunks
regardless of exact word overlap.

**Response Caching**
MD5-keyed in-memory cache with 1-hour TTL. Repeated queries
return instantly without re-running the LLM pipeline.

---

## What I Would Add Next
- Sentiment features from earnings calls to improve stock prediction
- Upgrade LLM to microsoft/phi-2 for better answer quality
- Switch to Redis for distributed caching
- Add user authentication
- Implement Prometheus monitoring
- Use IndexIVFFlat for million-scale vector search

---

## License
MIT