import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

def generate_mock_upi_data(output_file='upi_data.csv'):
    np.random.seed(42)
    start_date = datetime(2018, 1, 1)
    months = 72 # 6 years of monthly data
    
    dates = [start_date + relativedelta(months=i) for i in range(months)]
    
    # Base growth (exponential approximation for UPI)
    base_volume = np.exp(np.linspace(2, 8.5, months)) * 10 
    
    # Seasonality (Higher transactions typically around Oct/Nov festive season in India)
    seasonality = np.zeros(months)
    for i, d in enumerate(dates):
        if d.month in [10, 11]:
            seasonality[i] = base_volume[i] * 0.15 # 15% bump
        elif d.month in [1, 2]:
            seasonality[i] = -base_volume[i] * 0.05 # slight dip

    # Random noise
    noise = np.random.normal(0, base_volume * 0.05, months)
    
    volume_millions = base_volume + seasonality + noise
    # Ensure no negative values
    volume_millions = np.clip(volume_millions, a_min=10, a_max=None)
    
    df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in dates],
        'Volume_Millions': np.round(volume_millions, 2)
    })
    
    df.to_csv(output_file, index=False)
    print(f"Realistic mock UPI dataset generated and saved to {output_file}")

if __name__ == "__main__":
    generate_mock_upi_data()
