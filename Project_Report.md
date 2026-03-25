# Project Report: Enterprise UPI Forecasting & AI Analytics Engine

## 1. Project Title
**UPI Transaction Volume Forecasting and Optimization Engine**

## 2. Abstract
The Unified Payments Interface (UPI) has witnessed exponential growth in India, requiring banks and financial policymakers to aggressively scale their infrastructure. This project develops an end-to-end machine learning pipeline and interactive web dashboard to forecast future UPI transaction volumes, values, and participating banks. By pitting advanced `Auto-ARIMA` models against `Holt-Winters Exponential Smoothing`, the system provides highly accurate future projections. It also features automated anomaly detection and a dynamic "What-If" market shock simulator to evaluate systemic resilience. 

## 3. Technology Stack
* **Programming Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning & Time Series:** `pmdarima` (Auto-ARIMA), `statsmodels` (Holt-Winters, ADF Testing), `scikit-learn` (Model Evaluation)
* **Frontend Dashboard:** `Streamlit`
* **Interactive Visualizations:** `Plotly`

## 4. Methodology
The project follows a comprehensive Data Science lifecycle:

### A. Data Collection & Preprocessing
Historical UPI dataset containing metrics such as *Volume (in Mn)*, *Value (in Cr.)*, and *Active Banks* was processed. The month column was standardized into a datetime index, and the dataset was sorted chronologically to prepare for sequential time-series modeling.

### B. Exploratory Data Analysis (EDA) & Anomaly Detection
Before modeling, the data undergoes an algorithmic check. An automated **Rolling Z-Score** function constantly scans the historical timeline. Any data point deviating by more than 2.5 standard deviations from the rolling mean is flagged as a statistical outlier (Anomaly). This helps identifying sudden crashes or artificial spikes historically.

### C. Predictive Modeling (Ensemble approach)
The system employs an 80/20 Train-Test split validation strategy. Instead of relying on a single algorithm, the application simultaneously trains two competing financial models:
1. **Auto-ARIMA:** Automatically conducts grid-searches to find the optimal $(p, d, q)(P, D, Q)_m$ parameters, explicitly accounting for a 12-month seasonal lag.
2. **Holt-Winters Exponential Smoothing:** Serves as a robust baseline utilizing additive trend and seasonal components.

### D. Model Evaluation
Both models project values onto the unseen 20% test dataset. Their accuracy is quantified by calculating the **Mean Absolute Percentage Error (MAPE)**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**. The web engine dynamically highlights the model with the lowest MAPE as the 'Best Performing'.

## 5. Key Features of the Dashboard
* **Dynamic Metric Forecasting:** Users can independently forecast Volume, Value, or Bank count using the exact same underlying pipeline.
* **Confidence Intervals:** Generates strict 95% upper and lower statistical bounds on future predictions to represent modeling risk/variance.
* **What-If Scenario Simulation:** Users can inject a manual positive/negative shock percentage (e.g., +20% or -30%) to simulate a sudden macroeconomic policy change and observe how the final forecast target adapts.

## 6. Business Results & Strategic Insights
The engine produces three critical, actionable insights generated directly from the model outputs:

1. **Trend Identification:** The algorithm mathematically proves whether the underlying growth trajectory is Strongly Upward, Flat, or Downward.
2. **Insights for Banks:** Recommendations dictate that core banking server infrastructure must be expanded proactively to match the calculated baseline percentage growth. Furthermore, elastic cloud-computing readiness should be coupled directly to the *Upper Statistical Bound* to prevent 502 Gateway Timeouts during extreme seasonal (festive) spikes.
3. **Insights for Policy Makers:** The continuous upward mapping highlights deep systemic dependency on digital public infrastructure. The "What-If" engine is recommended for use in regulatory stress-testing to guarantee market liquidity during sudden shock events.
