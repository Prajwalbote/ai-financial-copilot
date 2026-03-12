"""
ui/app.py — Streamlit Frontend
===============================
WHY STREAMLIT?
    - Build professional UIs in pure Python
    - No HTML/CSS/JS knowledge needed
    - Perfect for AI/ML demos and internal tools
    - Used widely in the AI industry

WHAT YOU LEARN:
    - Streamlit components and layout
    - Session state management
    - Calling REST APIs from Python
    - Building multi-page apps
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="AI Financial Research Copilot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── API Configuration ────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"


# ── Helper Functions ─────────────────────────────────────────

def call_api(method: str, endpoint: str, data: dict = None) -> dict:
    """Make an API call and handle errors gracefully."""
    url = f"{API_BASE_URL}{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url, timeout=60)
        else:
            response = requests.post(url, json=data, timeout=120)

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False,
                "error": f"API Error {response.status_code}: {response.text}"
            }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to API. Make sure the API server is running."
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out. The model is still processing..."
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def check_api_health() -> bool:
    """Check if API is running."""
    result = call_api("GET", "/health")
    return result["success"]


# ── Custom CSS ───────────────────────────────────────────────
def apply_custom_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #4da6ff;
            text-align: center;
            padding: 1rem 0;
        }
        .sub-header {
            font-size: 1rem;
            color: #aaaaaa;
            text-align: center;
            margin-bottom: 2rem;
        }
        .answer-box {
            background-color: #1e3a5f;
            border-left: 4px solid #4da6ff;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
            color: #ffffff;
            font-size: 1rem;
            line-height: 1.6;
        }
        .metric-card {
            background-color: #1e1e2e;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            border: 1px solid #333;
        }
        .source-tag {
            background-color: #2d4a6e;
            color: #4da6ff;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            margin: 2px;
            display: inline-block;
        }
        .user-message {
            background-color: #2d2d2d;
            border-left: 4px solid #888;
            padding: 0.8rem;
            border-radius: 4px;
            margin: 0.5rem 0;
            color: #ffffff;
        }
    </style>
    """, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar with navigation and settings."""

    with st.sidebar:
        st.title("📈 Financial Copilot")
        st.markdown("---")

        # API Health Status
        st.subheader("System Status")
        if check_api_health():
            st.success("✅ API Online")
        else:
            st.error("❌ API Offline")
            st.warning(
                "Start the API:\n"
                "```\nuvicorn api.main:app --reload\n```"
            )

        st.markdown("---")

        # Index Selection
        st.subheader("Document Index")
        index_name = st.selectbox(
            "Select Index",
            options=["apple_10k", "documents", "batch"],
            help="Choose which document index to query"
        )

        st.markdown("---")

        st.subheader("About")
        st.markdown("""
        **AI Financial Research Copilot**

        Built with:
        - 🤗 HuggingFace Transformers
        - 🔍 FAISS Vector Search
        - ⚡ FastAPI Backend
        - 📊 Streamlit UI
        """)

    return index_name


# ── Page: Document Q&A ───────────────────────────────────────
def render_qa_page(index_name: str):
    """Render the document Q&A interface."""

    st.header("📄 Financial Document Q&A")
    st.markdown(
        "Ask questions about your ingested financial documents."
    )

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-message">🧑 {message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="answer-box">🤖 {message["content"]}</div>',
                unsafe_allow_html=True
            )
            if message.get("sources"):
                st.markdown("**Sources:**")
                for source in message["sources"]:
                    source_name = source.split("\\")[-1].split("/")[-1]
                    st.markdown(
                        f'<span class="source-tag">📄 {source_name}</span>',
                        unsafe_allow_html=True
                    )

    # Question input
    col1, col2 = st.columns([4, 1])

    with col1:
        question = st.text_input(
            "Ask a financial question",
            placeholder="What is Apple's total revenue for FY2023?",
            label_visibility="collapsed"
        )

    with col2:
        ask_button = st.button("Ask 🔍", use_container_width=True)

    # Quick question buttons
    st.markdown("**Quick Questions:**")
    quick_cols = st.columns(3)

    quick_questions = [
        "What are the total net sales?",
        "What is the operating income?",
        "How much was spent on R&D?",
    ]

    for i, qq in enumerate(quick_questions):
        if quick_cols[i].button(qq, use_container_width=True):
            question = qq
            ask_button = True

    # Process question
    if ask_button and question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.spinner("Searching documents and generating answer..."):
            result = call_api("POST", "/api/ask", {
                "question": question,
                "index_name": index_name,
                "top_k": 5
            })

        if result["success"]:
            data = result["data"]
            answer = data.get("answer", "No answer generated")
            sources = data.get("sources", [])

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Error: {result['error']}",
                "sources": []
            })

        st.rerun()

    # Summarize section
    st.markdown("---")
    st.subheader("📋 Document Summary")

    if st.button("Generate Summary", use_container_width=True):
        with st.spinner("Generating summary..."):
            result = call_api("POST", "/api/summarize", {
                "index_name": index_name
            })

        if result["success"]:
            summary = result["data"].get("summary", "")
            st.markdown(
                f'<div class="answer-box">{summary}</div>',
                unsafe_allow_html=True
            )
        else:
            st.error(result["error"])

    # Clear chat
    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


# ── Page: Stock Prediction ───────────────────────────────────
def render_prediction_page():
    """Render the stock prediction interface."""

    st.header("📈 Stock Price Prediction")
    st.markdown(
        "Predict next-day stock direction using ML and technical analysis."
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            placeholder="Enter ticker symbol",
            label_visibility="collapsed"
        ).upper()

    with col2:
        predict_button = st.button(
            "Predict 🔮",
            use_container_width=True
        )

    # Popular tickers
    st.markdown("**Popular Tickers:**")
    ticker_cols = st.columns(5)
    popular = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]

    for i, t in enumerate(popular):
        if ticker_cols[i].button(t, use_container_width=True):
            ticker = t
            predict_button = True

    if predict_button and ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            result = call_api("GET", f"/api/predict/{ticker}")

        if result["success"]:
            data = result["data"]
            direction = data.get("direction", "N/A")
            confidence = data.get("confidence", 0)
            price = data.get("current_price", 0)
            indicators = data.get("indicators", {})

            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                direction_emoji = "🔼" if direction == "UP" else "🔽"
                color = "green" if direction == "UP" else "red"
                st.markdown(
                    f'<div class="metric-card">'
                    f'<h2 style="color:{color}">'
                    f'{direction_emoji} {direction}</h2>'
                    f'<p style="color:#aaa">Predicted Direction</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence*100:.1f}%"
                )

            with col3:
                st.metric(
                    "Current Price",
                    f"${price:.2f}"
                )

            with col4:
                rsi = indicators.get("rsi", 0)
                signal = indicators.get("signal", "neutral")
                st.metric(
                    "RSI Signal",
                    f"{rsi:.1f}",
                    delta=signal
                )

            st.markdown("---")
            st.subheader("Technical Indicators")

            ind_col1, ind_col2 = st.columns(2)

            with ind_col1:
                st.markdown("**RSI Analysis**")
                rsi_val = indicators.get("rsi", 50)

                if rsi_val > 70:
                    st.warning(f"RSI: {rsi_val:.1f} — Overbought ⚠️")
                elif rsi_val < 30:
                    st.success(f"RSI: {rsi_val:.1f} — Oversold 💡")
                else:
                    st.info(f"RSI: {rsi_val:.1f} — Neutral ➡️")

                st.progress(rsi_val / 100)

            with ind_col2:
                st.markdown("**MACD Signal**")
                macd = indicators.get("macd", 0)
                if macd > 0:
                    st.success(f"MACD: {macd:.4f} — Bullish 📈")
                else:
                    st.error(f"MACD: {macd:.4f} — Bearish 📉")

        else:
            st.error(result["error"])


# ── Page: Risk Analysis ──────────────────────────────────────
def render_risk_page():
    """Render the risk analysis interface."""

    st.header("⚠️ Financial Risk Analysis")
    st.markdown(
        "Analyze risk metrics for any stock using institutional methods."
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        ticker = st.text_input(
            "Stock Ticker for Risk Analysis",
            value="AAPL",
            label_visibility="collapsed"
        ).upper()

    with col2:
        analyze_button = st.button(
            "Analyze Risk 📊",
            use_container_width=True
        )

    if analyze_button and ticker:
        with st.spinner(f"Calculating risk metrics for {ticker}..."):
            result = call_api("GET", f"/api/risk/{ticker}")

        if result["success"]:
            data = result["data"]
            risk_score = data.get("risk_score", 0)
            risk_level = data.get("risk_level", "UNKNOWN")
            metrics = data.get("metrics", {})
            interpretation = data.get("interpretation", "")

            st.markdown("---")
            col1, col2 = st.columns([1, 2])

            with col1:
                color_map = {
                    "LOW": "#28a745",
                    "MEDIUM": "#fd7e14",
                    "HIGH": "#dc3545"
                }
                color = color_map.get(risk_level, "gray")

                st.markdown(
                    f'<div class="metric-card">'
                    f'<h1 style="color:{color}">{risk_score:.0f}</h1>'
                    f'<h3 style="color:{color}">{risk_level} RISK</h3>'
                    f'<p style="color:#aaa">out of 100</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.progress(risk_score / 100)

            with col2:
                st.subheader("Risk Metrics")
                m1, m2 = st.columns(2)

                with m1:
                    st.metric(
                        "Annual Volatility",
                        f"{metrics.get('annual_volatility', 0):.1f}%"
                    )
                    st.metric(
                        "Max Drawdown",
                        f"{metrics.get('max_drawdown', 0):.1f}%"
                    )
                with m2:
                    st.metric(
                        "Value at Risk (95%)",
                        f"{metrics.get('value_at_risk_95', 0):.2f}%"
                    )
                    st.metric(
                        "Sharpe Ratio",
                        f"{metrics.get('sharpe_ratio', 0):.3f}"
                    )

            st.markdown("---")
            st.subheader("Risk Interpretation")
            st.markdown(
                f'<div class="answer-box">{interpretation}</div>',
                unsafe_allow_html=True
            )

            with st.expander("📚 What do these metrics mean?"):
                st.markdown("""
                **Annual Volatility** — How much the stock price
                swings in a year. Higher = more unpredictable.

                **Value at Risk (VaR 95%)** — The maximum daily
                loss you can expect with 95% confidence.

                **Sharpe Ratio** — Return per unit of risk.
                Above 1.0 is good. Below 0 means worse than
                a savings account.

                **Max Drawdown** — The biggest historical
                peak-to-trough price decline. Shows worst case.
                """)

        else:
            st.error(result["error"])


# ── Page: Document Ingestion ─────────────────────────────────
def render_ingest_page():
    """Render the document ingestion interface."""

    st.header("📥 Ingest Financial Documents")
    st.markdown(
        "Add new financial documents to the knowledge base."
    )

    tab1, tab2 = st.tabs(["Upload PDF", "Ingest URL"])

    with tab1:
        st.subheader("Upload a PDF Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload financial reports, earnings statements, etc."
        )

        index_name = st.text_input(
            "Index Name",
            value="documents",
            help="Name for this document collection"
        )

        if uploaded_file and st.button(
            "Upload and Process",
            use_container_width=True
        ):
            with st.spinner("Processing PDF..."):
                files = {"file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )}
                params = {"index_name": index_name}

                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/ingest/pdf",
                        files=files,
                        params=params,
                        timeout=120
                    )

                    if response.status_code == 200:
                        data = response.json()
                        st.success(
                            f"✅ Successfully processed! "
                            f"Created {data['chunks_created']} chunks"
                        )
                    else:
                        st.error(f"Error: {response.text}")

                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")

    with tab2:
        st.subheader("Ingest from URL")
        url = st.text_input(
            "Financial News URL",
            placeholder="https://finance.yahoo.com/news/..."
        )

        url_index = st.text_input(
            "Index Name",
            value="documents",
            key="url_index"
        )

        if url and st.button(
            "Fetch and Process",
            use_container_width=True
        ):
            with st.spinner(f"Fetching content from {url}..."):
                result = call_api("POST", "/api/ingest/url", {
                    "url": url,
                    "index_name": url_index
                })

            if result["success"]:
                data = result["data"]
                st.success(
                    f"✅ Successfully processed! "
                    f"Created {data['chunks_created']} chunks"
                )
            else:
                st.error(result["error"])


# ── Main App ─────────────────────────────────────────────────
def main():
    """Main application entry point."""

    apply_custom_css()

    # Header
    st.markdown(
        '<div class="main-header">📈 AI Financial Research Copilot</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">'
        'Powered by RAG + LLM + ML | Built with HuggingFace & FAISS'
        '</div>',
        unsafe_allow_html=True
    )

    # Sidebar
    index_name = render_sidebar()

    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Document Q&A",
        "📈 Stock Prediction",
        "⚠️ Risk Analysis",
        "📥 Ingest Documents"
    ])

    with tab1:
        render_qa_page(index_name)

    with tab2:
        render_prediction_page()

    with tab3:
        render_risk_page()

    with tab4:
        render_ingest_page()


if __name__ == "__main__":
    main()