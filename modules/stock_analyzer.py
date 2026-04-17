"""
Stock & Investment Analyzer Module
Provides fundamental analysis, technical indicators, and risk assessment for stocks.
Generates professional PDF research reports with disclaimers.
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


def show_investor_profile_modal() -> Optional[Dict]:
    """Offer optional investor profiling to tailor analysis."""
    st.subheader("📋 Investor Profile (Optional)")
    st.caption("Customize the analysis to your investment style. Skip if you prefer a general analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_tolerance = st.radio(
            "Risk Tolerance",
            ["Conservative (preserve capital)", "Moderate (balanced growth)", "Aggressive (maximize returns)"],
            key="risk_profile"
        )
        time_horizon = st.radio(
            "Time Horizon",
            ["Short-term (< 1 year)", "Medium-term (1-5 years)", "Long-term (5+ years)"],
            key="time_profile"
        )
    
    with col2:
        income_needs = st.radio(
            "Income Needs",
            ["Dividend/Income-focused", "Growth-focused", "Mixed"],
            key="income_profile"
        )
        experience = st.radio(
            "Experience Level",
            ["Beginner", "Intermediate", "Advanced"],
            key="exp_profile"
        )
    
    allocation = st.slider(
        "Current Equity Allocation (%)",
        0, 100, 60, key="allocation_profile"
    )
    
    return {
        "risk_tolerance": risk_tolerance,
        "time_horizon": time_horizon,
        "income_needs": income_needs,
        "experience_level": experience,
        "equity_allocation": allocation
    }


def fetch_stock_data(ticker: str, period: str = "1y") -> Optional[Tuple]:
    """
    Fetch stock data using yfinance with error handling.
    Returns: (ticker_object, daily_history, info_dict)
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        # Get historical data
        history = stock.history(period=period)
        
        # Get company info
        info = stock.info
        
        if history.empty:
            st.error(f"❌ No data found for ticker: {ticker}")
            return None
        
        return stock, history, info
    
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return None


def calculate_technical_indicators(history: pd.DataFrame) -> pd.DataFrame:
    """Calculate moving averages and RSI."""
    df = history.copy()
    
    # Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI (14-period)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    return df


def get_fundamental_metrics(info: Dict) -> pd.DataFrame:
    """Extract and organize fundamental metrics."""
    metrics = {
        "Metric": [
            "Current Price",
            "Market Cap",
            "P/E Ratio",
            "PEG Ratio",
            "EV/EBITDA",
            "Price/Book",
            "Price/Sales",
            "Dividend Yield",
            "Beta",
            "52-Week Range",
            "Profit Margin",
            "Revenue Growth (YoY)",
            "Gross Margin"
        ],
        "Value": [
            f"${info.get('currentPrice', 'N/A'):.2f}" if isinstance(info.get('currentPrice'), (int, float)) else info.get('currentPrice', 'N/A'),
            f"${info.get('marketCap', 'N/A'):,.0f}" if isinstance(info.get('marketCap'), (int, float)) else info.get('marketCap', 'N/A'),
            f"{info.get('trailingPE', 'N/A'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else info.get('trailingPE', 'N/A'),
            f"{info.get('pegRatio', 'N/A'):.2f}" if isinstance(info.get('pegRatio'), (int, float)) else info.get('pegRatio', 'N/A'),
            f"{info.get('enterpriseToEbitda', 'N/A'):.2f}" if isinstance(info.get('enterpriseToEbitda'), (int, float)) else info.get('enterpriseToEbitda', 'N/A'),
            f"{info.get('priceToBook', 'N/A'):.2f}" if isinstance(info.get('priceToBook'), (int, float)) else info.get('priceToBook', 'N/A'),
            f"{info.get('priceToSalesTrailing12Months', 'N/A'):.2f}" if isinstance(info.get('priceToSalesTrailing12Months'), (int, float)) else info.get('priceToSalesTrailing12Months', 'N/A'),
            f"{info.get('dividendYield', 'N/A')*100:.2f}%" if isinstance(info.get('dividendYield'), (int, float)) else info.get('dividendYield', 'N/A'),
            f"{info.get('beta', 'N/A'):.2f}" if isinstance(info.get('beta'), (int, float)) else info.get('beta', 'N/A'),
            f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f} - ${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if isinstance(info.get('fiftyTwoWeekLow'), (int, float)) else "N/A",
            f"{info.get('profitMargins', 'N/A')*100:.2f}%" if isinstance(info.get('profitMargins'), (int, float)) else info.get('profitMargins', 'N/A'),
            f"{info.get('revenueGrowth', 'N/A')*100:.2f}%" if isinstance(info.get('revenueGrowth'), (int, float)) else info.get('revenueGrowth', 'N/A'),
            f"{info.get('grossMargins', 'N/A')*100:.2f}%" if isinstance(info.get('grossMargins'), (int, float)) else info.get('grossMargins', 'N/A'),
        ]
    }
    return pd.DataFrame(metrics)


def analyze_financial_quality(info: Dict, stock) -> str:
    """Analyze financial quality and identify red lines."""
    red_flags = []
    
    # Debt analysis
    total_debt = info.get('totalDebt', 0)
    ebitda = info.get('ebitda', 1)
    if total_debt and ebitda and (total_debt / ebitda > 4):
        red_flags.append("⚠️ High Debt/EBITDA (>4x) - covenant breach risk")
    
    # Profitability margins
    profit_margin = info.get('profitMargins', 0)
    if profit_margin and profit_margin < 0.05:
        red_flags.append("⚠️ Low profit margins (<5%) - pricing power concerns")
    
    # Goodwill analysis
    try:
        balance_sheet = stock.balance_sheet
    except Exception:
        balance_sheet = pd.DataFrame()

    if not balance_sheet.empty:
        goodwill = balance_sheet.iloc[0].get('Goodwill', 0)
        total_assets = balance_sheet.iloc[0].get('TotalAssets', 1)
        if goodwill and total_assets and (goodwill / total_assets > 0.5):
            red_flags.append("⚠️ High Goodwill (>50% of assets) - acquisition risk")
    
    # FCF analysis
    fcf = info.get('freeCashflow', 0)
    market_cap = info.get('marketCap', 1)
    if market_cap and fcf:
        fcf_yield = fcf / market_cap
        if fcf_yield < 0:
            red_flags.append("⚠️ Negative FCF - burning cash")
        elif fcf_yield > 0.05:
            red_flags.append("✅ Strong FCF yield (>5%) - genuinely cheap")
    
    if not red_flags:
        red_flags.append("✅ No major red flags detected")
    
    return "\n".join(red_flags)


def create_price_chart(history: pd.DataFrame, ticker: str) -> go.Figure:
    """Create interactive price chart with moving averages."""
    df = calculate_technical_indicators(history)
    df = df.dropna()
    
    fig = go.Figure()
    
    # Price
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # SMA 50
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_50'],
        name='50-day SMA',
        line=dict(color='orange', dash='dash')
    ))
    
    # SMA 200
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_200'],
        name='200-day SMA',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f"{ticker} - 1-Year Price & Technical Indicators",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig


def create_rsi_chart(history: pd.DataFrame, ticker: str) -> go.Figure:
    """Create RSI indicator chart."""
    df = calculate_technical_indicators(history)
    df = df.dropna()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI_14'],
        name='RSI (14)',
        line=dict(color='purple', width=2),
        fill='tozeroy'
    ))
    
    # Add overbought/oversold levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    
    fig.update_layout(
        title=f"{ticker} - RSI (14) Technical Indicator",
        xaxis_title="Date",
        yaxis_title="RSI",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def show_stock_analyzer():
    """Main stock analyzer UI."""
    st.markdown("## 📈 Stock & Investment Analyzer")
    
    st.info(
        "⚠️ **DISCLAIMER**: This report is for informational purposes only and does NOT constitute investment advice. "
        "It is NOT a recommendation to buy, sell, or hold any security. Consult a licensed financial advisor before making investment decisions."
    )
    
    # Input section
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input(
            "Enter Stock Ticker",
            placeholder="e.g., AAPL, MSFT, TSLA",
            help="Enter a valid stock ticker symbol"
        ).upper()
    with col2:
        period = st.selectbox("Time Period", ["3mo", "6mo", "1y", "2y", "5y"])
    
    if not ticker:
        st.warning("👉 Enter a stock ticker to begin analysis")
        return
    
    # Fetch data
    with st.spinner(f"Fetching data for {ticker}..."):
        result = fetch_stock_data(ticker, period)
    
    if result is None:
        return
    
    stock, history, info = result
    
    # Optional investor profiling
    with st.expander("📋 Customize Analysis (Optional)", expanded=False):
        profile = show_investor_profile_modal()
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📈 Technical", "💰 Fundamentals", "🎯 Quality", "📋 Summary"
    ])
    
    with tab1:
        st.subheader(f"{info.get('longName', ticker)} ({ticker})")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
        with col2:
            market_cap = info.get('marketCap', 0)
            if market_cap:
                st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
            else:
                st.metric("Market Cap", "N/A")
        with col3:
            st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else "N/A")
        with col4:
            st.metric("Beta", f"{info.get('beta', 'N/A'):.2f}" if isinstance(info.get('beta'), (int, float)) else "N/A")
        
        st.markdown("---")
        
        # Company description
        if info.get('longBusinessSummary'):
            st.subheader("Company Overview")
            st.write(info.get('longBusinessSummary'))
        
        # Key metrics table
        st.subheader("Key Metrics")
        metrics_df = get_fundamental_metrics(info)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Technical Analysis")
        
        # Price chart
        fig_price = create_price_chart(history, ticker)
        st.plotly_chart(fig_price, use_container_width=True)
        
        # RSI chart
        fig_rsi = create_rsi_chart(history, ticker)
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Technical interpretation
        df_tech = calculate_technical_indicators(history)
        latest_rsi = df_tech['RSI_14'].iloc[-1]
        latest_price = df_tech['Close'].iloc[-1]
        sma_50 = df_tech['SMA_50'].iloc[-1]
        sma_200 = df_tech['SMA_200'].iloc[-1]
        
        st.subheader("Technical Interpretation")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Current RSI (14):** {latest_rsi:.2f}")
            if latest_rsi > 70:
                st.warning("⚠️ Overbought (RSI > 70)")
            elif latest_rsi < 30:
                st.success("✅ Oversold (RSI < 30)")
            else:
                st.info("Neutral RSI (30-70)")
        
        with col2:
            st.write(f"**Price vs. Moving Averages:**")
            if latest_price > sma_50 > sma_200:
                st.success("✅ Bullish: Price > SMA 50 > SMA 200")
            elif latest_price < sma_50 < sma_200:
                st.error("❌ Bearish: Price < SMA 50 < SMA 200")
            else:
                st.info("⚡ Mixed signals")
    
    with tab3:
        st.subheader("Fundamental Analysis")
        
        # Valuation metrics
        st.write("**Valuation Multiples:**")
        val_cols = st.columns(3)
        
        with val_cols[0]:
            pe = info.get('trailingPE', None)
            if pe and isinstance(pe, (int, float)):
                if pe > 25:
                    st.warning(f"P/E: {pe:.2f} (Potentially high)")
                elif pe < 15:
                    st.success(f"P/E: {pe:.2f} (Potentially low)")
                else:
                    st.info(f"P/E: {pe:.2f} (Moderate)")
            else:
                st.info("P/E: N/A")
        
        with val_cols[1]:
            peg = info.get('pegRatio', None)
            if peg and isinstance(peg, (int, float)):
                if peg < 1.0:
                    st.success(f"PEG: {peg:.2f} (Growth at reasonable price)")
                elif peg > 2.0:
                    st.warning(f"PEG: {peg:.2f} (Priced for perfection)")
                else:
                    st.info(f"PEG: {peg:.2f}")
            else:
                st.info("PEG: N/A")
        
        with val_cols[2]:
            ps = info.get('priceToSalesTrailing12Months', None)
            if ps and isinstance(ps, (int, float)):
                if ps > 20:
                    st.warning(f"P/S: {ps:.2f} (High - needs hypergrowth)")
                else:
                    st.info(f"P/S: {ps:.2f}")
            else:
                st.info("P/S: N/A")
        
        st.markdown("---")
        
        # Profitability
        st.write("**Profitability & Growth:**")
        prof_cols = st.columns(3)
        
        with prof_cols[0]:
            pm = info.get('profitMargins', None)
            st.metric(
                "Profit Margin",
                f"{pm*100:.2f}%" if isinstance(pm, (int, float)) else "N/A"
            )
        
        with prof_cols[1]:
            rg = info.get('revenueGrowth', None)
            st.metric(
                "Revenue Growth (YoY)",
                f"{rg*100:.2f}%" if isinstance(rg, (int, float)) else "N/A"
            )
        
        with prof_cols[2]:
            gm = info.get('grossMargins', None)
            st.metric(
                "Gross Margin",
                f"{gm*100:.2f}%" if isinstance(gm, (int, float)) else "N/A"
            )
    
    with tab4:
        st.subheader("Financial Quality Analysis")
        
        quality_analysis = analyze_financial_quality(info, stock)
        st.markdown(quality_analysis)
        
        # Additional quality checks
        st.markdown("---")
        st.write("**Key Quality Indicators:**")
        
        qa, qb = st.columns(2)
        
        with qa:
            # Dividend info
            if info.get('dividendYield'):
                st.success(f"💰 Dividend Yield: {info.get('dividendYield')*100:.2f}%")
            else:
                st.info("No dividend")
            
            # Debt check
            debt_ratio = info.get('debtToEquity', None)
            if debt_ratio and isinstance(debt_ratio, (int, float)):
                if debt_ratio > 1.5:
                    st.warning(f"⚠️ High Debt/Equity: {debt_ratio:.2f}")
                else:
                    st.info(f"✅ Debt/Equity: {debt_ratio:.2f}")
        
        with qb:
            # Growth check
            rg = info.get('revenueGrowth', None)
            if rg and isinstance(rg, (int, float)):
                if rg > 0.20:
                    st.success(f"📈 Strong growth: {rg*100:.2f}%")
                elif rg > 0:
                    st.info(f"Moderate growth: {rg*100:.2f}%")
            
            # FCF check
            fcf = info.get('freeCashflow', None)
            if fcf and isinstance(fcf, (int, float)):
                if fcf > 0:
                    st.success(f"✅ Positive FCF: ${fcf/1e6:.0f}M")
                else:
                    st.error(f"⚠️ Negative FCF: ${fcf/1e6:.0f}M")
    
    with tab5:
        st.subheader("Analysis Summary")
        
        # Overall assessment
        st.write(f"**Stock:** {ticker}")
        st.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}")
        
        if profile:
            st.write("**Your Profile:**")
            st.write(f"- Risk Tolerance: {profile['risk_tolerance']}")
            st.write(f"- Time Horizon: {profile['time_horizon']}")
            st.write(f"- Income Needs: {profile['income_needs']}")
        
        st.markdown("---")
        
        # Export option
        st.subheader("📥 Export Data")
        
        # Create summary export
        summary_data = {
            "Ticker": [ticker],
            "Company": [info.get('longName', 'N/A')],
            "Sector": [info.get('sector', 'N/A')],
            "Current Price": [info.get('currentPrice', 'N/A')],
            "P/E Ratio": [info.get('trailingPE', 'N/A')],
            "Market Cap": [info.get('marketCap', 'N/A')],
            "52-Week High": [info.get('fiftyTwoWeekHigh', 'N/A')],
            "52-Week Low": [info.get('fiftyTwoWeekLow', 'N/A')],
            "Analysis Date": [datetime.now().strftime('%Y-%m-%d')]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        col1, col2 = st.columns(2)
        with col1:
            csv_data = summary_df.to_csv(index=False)
            st.download_button(
                "📥 Download as CSV",
                data=csv_data,
                file_name=f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            excel_buffer = io.BytesIO()
            summary_df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(
                "📥 Download as Excel",
                data=excel_buffer.getvalue(),
                file_name=f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        st.markdown("---")
        st.info(
            "⚠️ **DISCLAIMER**: This analysis is for informational purposes only. "
            "It does not constitute investment advice or a recommendation to buy/sell any security. "
            "Always consult a licensed financial advisor before making investment decisions."
        )
