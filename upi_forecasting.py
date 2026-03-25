import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')

# Set plotting styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

def load_data(filepath='upi_data.csv'):
    """Load data and set datetime index."""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Please run data_generator.py first to create the dataset.")
        return None
    
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def perform_eda(df):
    """Plot historical data and seasonal decomposition."""
    print("Performing Exploratory Data Analysis (EDA)...")
    
    # Plot historical transactions
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Volume_Millions'], linewidth=2)
    plt.title('Historical UPI Transaction Volume (Millions)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Volume (Millions)')
    plt.tight_layout()
    plt.savefig('eda_historical_volume.png', dpi=300)
    plt.close()
    
    # Perform time series decomposition
    try:
        decomposition = seasonal_decompose(df['Volume_Millions'], model='multiplicative', period=12)
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        plt.suptitle('Time Series Decomposition (Multiplicative)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('eda_decomposition.png', dpi=300)
        plt.close()
        print("EDA plots saved: eda_historical_volume.png, eda_decomposition.png")
    except Exception as e:
        print(f"Decomposition warning: {e}")

def test_stationarity(timeseries):
    """Check stationarity using Augmented Dickey-Fuller test."""
    print("\nRunning Augmented Dickey-Fuller Test for Stationarity...")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)
    
    p_value = dfoutput['p-value']
    if p_value < 0.05:
        print(f"p-value ({p_value:.4f}) < 0.05. The time series is stationary.")
    else:
        print(f"p-value ({p_value:.4f}) > 0.05. The time series is NOT stationary. Differencing is required.")
        
    return p_value < 0.05

def train_evaluate_arima(df):
    """Train ARIMA model using auto_arima and evaluate on test set."""
    print("\nSplitting data into 80% Train / 20% Test sets...")
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    print("Finding optimal ARIMA parameters (p,d,q)(P,D,Q) using auto_arima based on AIC...")
    # auto_arima will iterate & evaluate different combinations
    model = auto_arima(train['Volume_Millions'], 
                       m=12,            # Annual seasonality (12 months in a year)
                       seasonal=True,   # Data exhibits yearly seasonality
                       d=None,          # Let the model algorithm determine the optimal differencing 'd'
                       trace=True,      # Print progress
                       error_action='ignore',  
                       suppress_warnings=True, 
                       stepwise=True)   # Use stepwise algorithm for faster search
                       
    print("\nBest Model Found:")
    print(model.summary())
    
    print("\nEvaluating model against test set...")
    predictions = model.predict(n_periods=len(test))
    
    mae = mean_absolute_error(test['Volume_Millions'], predictions)
    mse = mean_squared_error(test['Volume_Millions'], predictions)
    rmse = np.sqrt(mse)
    
    print(f"Model Performance Metrics:")
    print(f"Mean Absolute Error (MAE):     {mae:,.2f} Millions")
    print(f"Mean Squared Error (MSE):      {mse:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} Millions")
    
    # Plot evaluation
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Volume_Millions'], label='Train Data')
    plt.plot(test.index, test['Volume_Millions'], label='Test Data (Actual)')
    plt.plot(test.index, predictions, label='ARIMA Predictions', color='red', linestyle='dashed')
    plt.title('ARIMA Model Evaluation (Train, Test & Predictions)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Volume (Millions)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('result_evaluation_plot.png', dpi=300)
    plt.close()
    print("Evaluation plot saved: result_evaluation_plot.png")
    
    return model

def forecast_future(model, df, horizon=12):
    """Forecast future UPI volumes and return details."""
    print(f"\nForecasting for the next {horizon} months...")
    future_forecast = model.predict(n_periods=horizon)
    
    last_date = df.index[-1]
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted_Volume': future_forecast
    }).set_index('Date')
    
    print("\nForecasted Monthly Volumes (Next 12 Months):")
    print(forecast_df)
    
    # Plot Future Forecast
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Volume_Millions'], label='Historical Actuals', linewidth=2)
    plt.plot(forecast_df.index, forecast_df['Forecasted_Volume'], label='Future Forecast', color='orange', linestyle='--', linewidth=2)
    
    # Highlight final data point and end of forecast point
    plt.scatter(df.index[-1], df['Volume_Millions'].iloc[-1], color='blue', zorder=5)
    plt.scatter(forecast_df.index[-1], forecast_df['Forecasted_Volume'].iloc[-1], color='orange', zorder=5)
    
    plt.title('Future UPI Transaction Volume Forecast (12 Months)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Volume (Millions)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('result_future_forecast.png', dpi=300)
    plt.close()
    print("Forecast plot saved: result_future_forecast.png")
    
    return forecast_df

def generate_insights(df, forecast_df):
    """Generates an interpretation and action report."""
    current_vol = df['Volume_Millions'].iloc[-1]
    future_vol = forecast_df['Forecasted_Volume'].iloc[-1]
    growth_pct = ((future_vol - current_vol) / current_vol) * 100
    avg_monthly_growth = forecast_df['Forecasted_Volume'].pct_change().mean() * 100
    
    report = f"""
===================================================================
              UPI TRANSACTION FORECASTING INSIGHTS
===================================================================
1. Overall Trajectory Assessment:
   - Historical Context: The UPI network volumes have demonstrated a 
     rapid, exponential long-term adoption curve.
   - Current Volume (Last Recorded): {current_vol:,.2f} Million transactions.
   - 12-Month Forecasted Target:   {future_vol:,.2f} Million transactions.
   - Estimated Annual Growth Rate: {growth_pct:.2f}%

2. Seasonality & Volatility Insights:
   - Recurrent peaks identify specific calendar windows (e.g., Oct-Nov 
     festive season in India) which historically drive disproportionate volume.
   - The ARIMA model has successfully captured this multiplicative 
     seasonal lag, projecting corresponding cyclical surges in the 
     upcoming year.

3. Actionable Recommendations for Banks & Policymakers:
   - CAPACITY PLANNING: Core banking systems should prep scale to handle 
     a baseline increase of ~{int(growth_pct)}% over the next 12 months.
   - FESTIVE SEASON READINESS: Critical system freeze and server capacity 
     scaling must be finalized by September to prevent downtime 
     during projected Q4 surges.
   - AVERAGE MONTHLY EXPANSION: On average, server loads will increase 
     by {avg_monthly_growth:.2f}% month-over-month. Consistent cloud elasticity tuning is advised.
===================================================================
"""
    with open('insights_report.txt', 'w') as f:
        f.write(report)
        
    print(report)
    print("Insights report written to insights_report.txt")

def main():
    print("====================================")
    print(" UPI TRANSACTION FORECASTING SCRIPT ")
    print("====================================\n")
    
    # 1. Load Data
    df = load_data()
    if df is None: return
    
    # 2 & 3. Exploratory Data Analysis & Time Series Components
    perform_eda(df)
    
    # Time Series Stationarity Check before Modeling
    is_stationary = test_stationarity(df['Volume_Millions'])
    
    # 4, 5, 7. Develop, Train, & Evaluate ARIMA Model
    model = train_evaluate_arima(df)
    
    # 6. Forecast Future Volumes
    forecast_df = forecast_future(model, df)
    
    # 8 & 9. Interpret Results and Provide Insights
    generate_insights(df, forecast_df)
    
    print("\n[SUCCESS] Entire workflow completed. Check the current directory for results!")

if __name__ == "__main__":
    main()
