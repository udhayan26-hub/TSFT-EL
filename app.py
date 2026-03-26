import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
import os

warnings.filterwarnings('ignore')

# Premium Dashboard Configuration
st.set_page_config(page_title="UPI Analytics Hub", layout="wide", page_icon="🚀")

# Modern UI Styles via HTML/CSS Customization
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #F39C12;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #F39C12;
    }
    .metric-label {
        color: #A6ACAF;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(filepath='upi_data_enhanced.csv'):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Month'], format='%b-%y')
    df = df.sort_values(by='Date')
    df.set_index('Date', inplace=True)
    return df

@st.cache_resource
def train_arima(train_data):
    # Optimizing the ARIMA search space to prevent the frontend from hanging
    model = auto_arima(
        train_data, 
        m=12, 
        seasonal=True, 
        trace=False, 
        error_action='ignore', 
        suppress_warnings=True, 
        stepwise=True,
        max_p=2, 
        max_q=2, 
        max_P=1, 
        max_Q=1, 
        max_D=1
    )
    return model

@st.cache_resource
def train_hw(train_data):
    model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12).fit(optimized=True)
    return model

def detect_anomalies(series, window=6, threshold=2.5):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    anomalies = series[np.abs(series - rolling_mean) > (threshold * rolling_std)]
    return anomalies

# ==================== MAIN UI ====================
st.title("🚀 Enterprise UPI Analytics Engine")
st.markdown("*Advanced predictive intelligence designed for executive decision-making, automated anomaly tracking, and risk simulation.*")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Please upload `upi_data_enhanced.csv`")
    st.stop()

# Interactive Scenario Parameters (Sidebar)
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_container_width=True)
elif os.path.exists("logo.jpg"):
    st.sidebar.image("logo.jpg", use_container_width=True)
else:
    st.sidebar.caption("Please save your uploaded logo as 'logo.png' in the main folder to display it in this sidebar!")

st.sidebar.title("Hyperparameters")
st.sidebar.markdown("---")

# Target Metric Selection
target_feature = st.sidebar.selectbox("🎯 Target Metric to Forecast", 
                                      ['Volume (in Mn)', 'Value (in Cr.)', 'No. of Banks live on UPI'])

forecast_horizon = st.sidebar.slider("📅 Forecast Horizon (Months)", min_value=3, max_value=48, value=12, help="How many months into the future to project.")
train_split = st.sidebar.slider("🧪 Machine Learning Train/Test Split %", 60, 95, 80, help="Percentage of data strictly used for training.")

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ What-If Risk Simulation")
shock_impact = st.sidebar.slider("Simulate Market Shock (%)", -50, 50, 0, help="Simulate a sudden policy change, global event, or technological breakthrough immediately impacting future projections.")

# Extract time series data
ts_data = df[target_feature].dropna()

# Detect Anomalies prior to modeling
anomalies = detect_anomalies(ts_data)

# Splitting Data
train_size = int(len(ts_data) * (train_split / 100))
train, test = ts_data.iloc[:train_size], ts_data.iloc[train_size:]

# Train Models
with st.spinner(f"Training Enterprise Ensemble on {target_feature}..."):
    arima_model = train_arima(train)
    hw_model = train_hw(train)

# Predict Test values
arima_test_preds = arima_model.predict(n_periods=len(test))
hw_test_preds = hw_model.forecast(len(test))

# Advanced Metric: Mean Absolute Percentage Error (MAPE)
arima_mape = mean_absolute_percentage_error(test, arima_test_preds) * 100
hw_mape = mean_absolute_percentage_error(test, hw_test_preds) * 100

st.divider()

# --- TABS FOR ORGANIZED PRESENTATION ---
tab1, tab2, tab3, tab4 = st.tabs(["📂 Dataset Overview", "📊 Diagnostic Analytics", "🤖 Model Competition", "🔮 Prescriptive Forecasting"])

with tab1:
    st.subheader("Raw Dataset Overview & Descriptive Statistics")
    st.markdown("Explore the core foundational data before analytical modeling is applied. The complete dataset contains historical UPI operations tracked monthly.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Complete Time-Series Data**")
        st.dataframe(df, use_container_width=True, height=400)
        
    with col2:
        st.markdown(f"**Statistical Summary: `{target_feature}`**")
        st.dataframe(ts_data.describe(), use_container_width=True)
        st.info(f"Loaded **{len(df)}** total monthly records spanning from **{df.index.min().strftime('%b %Y')}** to **{df.index.max().strftime('%b %Y')}**.")

with tab2:
    st.subheader(f"Historical Trajectory & Automated Anomaly Detection: {target_feature}")
    fig_hist = go.Figure()
    
    # Baseline
    fig_hist.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines', name='Actual Network Data', line=dict(color='#3498DB', width=3)))
    
    # Draw Anomalies
    if not anomalies.empty:
        fig_hist.add_trace(go.Scatter(x=anomalies.index, y=anomalies.values, mode='markers', name='Detected Statistical Outliers', 
                                      marker=dict(color='#E74C3C', size=12, symbol='x', line=dict(width=2, color='white'))))
    
    fig_hist.update_layout(height=450, xaxis_title="Timeline", yaxis_title=target_feature, template="plotly_dark",
                           hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_hist, use_container_width=True)
    
    colA, colB = st.columns(2)
    colA.info(f"**Total Verified Monthly Records:** {len(ts_data)}")
    colB.warning(f"**Anomalies Isolated (Z-Score Threshold):** {len(anomalies)} Extreme Growth/Crash Events Pinpointed")

with tab3:
    st.subheader("Model Validation: Auto-ARIMA vs. Exponential Smoothing (Holt-Winters)")
    st.markdown("We pit traditional exponential smoothing against advanced Auto-ARIMA predictive components. *Lower MAPE (Mean Absolute Percentage Error) wins.*")
    
    c1, c2, c3 = st.columns(3)
    best_model = "Auto-ARIMA 🏆" if arima_mape < hw_mape else "Holt-Winters 🏆"
    
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Best Performing Algorithm</div><div class="metric-value">{best_model}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">ARIMA Error (MAPE)</div><div class="metric-value">{arima_mape:.2f}%</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Holt-Winters Error (MAPE)</div><div class="metric-value">{hw_mape:.2f}%</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    fig_eval = go.Figure()
    fig_eval.add_trace(go.Scatter(x=train.index, y=train.values, mode='lines', name='Training Baseline Phase', line=dict(color='#7F8C8D', width=2)))
    fig_eval.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines', name='Actual Future (Truth Dataset)', line=dict(color='#2ECC71', width=4)))
    fig_eval.add_trace(go.Scatter(x=test.index, y=arima_test_preds, mode='lines', name='ARIMA Engine Prediction', line=dict(color='#E74C3C', dash='dash', width=2)))
    fig_eval.add_trace(go.Scatter(x=test.index, y=hw_test_preds, mode='lines', name='Holt-Winters Engine Prediction', line=dict(color='#9B59B6', dash='dot', width=2)))
    
    fig_eval.update_layout(height=500, template="plotly_dark", hovermode="x unified", title=f"Holdout Set Testing (Simulating the unseen {100-train_split}%)")
    st.plotly_chart(fig_eval, use_container_width=True)

with tab4:
    st.subheader(f"Future Scenario Planning & Risk Assessment ({forecast_horizon} Months)")
    
    with st.spinner("Executing final ensemble forecasting on target horizon..."):
        from pmdarima.arima import ARIMA
        # We always forecast the future using ALL historical data
        # To avoid the slow auto_arima search, we reuse the optimal parameters found during training
        final_arima = ARIMA(order=arima_model.order, seasonal_order=arima_model.seasonal_order, suppress_warnings=True)
        final_arima.fit(ts_data)
        future_forecast, conf_int = final_arima.predict(n_periods=forecast_horizon, return_conf_int=True)
        
        # Inject Shock Impact if modified by User
        if shock_impact != 0:
            multiplier = 1 + (shock_impact / 100)
            future_forecast *= multiplier
            conf_int *= multiplier

    last_date = ts_data.index[-1]
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, forecast_horizon + 1)]
    
    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines', name='Historical Trajectory', line=dict(color='#3498DB', width=3)))
    
    scenario_name = f'What-If Altered Forecast (Shock: {shock_impact}%)' if shock_impact != 0 else 'Standard Baseline Forecast'
    shock_color = '#F1C40F' if shock_impact == 0 else '#E74C3C'
    
    fig_future.add_trace(go.Scatter(x=future_dates, y=future_forecast, mode='lines', name=scenario_name, line=dict(color=shock_color, width=4)))
    
    # Confidence Intervals
    fig_future.add_trace(go.Scatter(
        name='Stat. Upper Bound (+95%)', x=future_dates, y=conf_int[:, 1],
        mode='lines', marker=dict(color="#444"), line=dict(width=0), showlegend=False
    ))
    fig_future.add_trace(go.Scatter(
        name='Stat. Lower Bound (-95%)', x=future_dates, y=conf_int[:, 0],
        marker=dict(color="#444"), line=dict(width=0), mode='lines', fillcolor='rgba(241, 196, 15, 0.1)', fill='tonexty', showlegend=False
    ))

    fig_future.update_layout(height=600, template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_future, use_container_width=True)
    
    # Generated Business Prescriptive Insight
    current_vol = ts_data.iloc[-1]
    future_vol = float(future_forecast.iloc[-1])
    growth_pct = ((future_vol - current_vol) / current_vol) * 100
    
    # Determine the trend direction dynamically
    if growth_pct > 5:
        trend_direction = "Strongly Upward 📈"
    elif growth_pct > 0:
        trend_direction = "Moderately Upward ↗️"
    elif growth_pct > -5:
        trend_direction = "Relatively Flat / Stable ➡️"
    else:
        trend_direction = "Downward 📉"
        
    st.markdown("### 🧠 Interpretation & Strategic Insights")
    
    st.info(f"""
    **1. Trend Interpretation**
    The mathematical formulation models a **{trend_direction}** trajectory for `{target_feature}`. Over the selected {forecast_horizon}-month horizon, the recorded metric is statistically projected to move from **{current_vol:,.0f}** to **{future_vol:,.0f}**, achieving a cumulative vector shift of **{growth_pct:+.2f}%**.
    """)
    
    st.success(f"""
    **2. Actionable Insights for Banks**
    - **Infrastructure Scaling:** To prevent server overloads, gateway timeouts, and 502 errors, core banking infrastructure parameters must be expanded proactively to confidently handle the anticipated **{max(0, int(growth_pct))}%** baseline growth.
    - **Capacity Buffering:** The {best_model.replace('🏆','').strip()} model's 95% confidence variance indicates potentially extreme seasonal volatility; rapid elastic server allocation is highly recommended immediately preceding festive bottleneck cycles.

    **3. Actionable Insights for Policy Makers**
    - **Economic & Digital Trajectory:** The fundamental {trend_direction.split(' ')[0].lower()} trend forcefully underscores broadening national digital penetration and heightened consumer reliance on digital public goods.
    - **Systemic Risk Mitigation:** Simulating a **{shock_impact}%** market shock dynamically stress-tests the model positioning the end-target at **{future_vol:,.0f}**. Institutional regulatory frameworks and financial liquidity policies must continuously be validated against these upper and lower statistical bounds to guarantee unwavering macroeconomic stability.
    """)
