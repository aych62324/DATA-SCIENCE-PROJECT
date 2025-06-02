import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

data = pd.read_csv('data2.csv')
data_2024 = data[data['Year'] == 2024].copy()

month_order = ['january', 'february', 'march', 'april', 'may', 'june', 
               'july', 'august', 'september', 'october', 'november', 'december']
data_2024['month_ID'] = data_2024['Month'].apply(lambda x: month_order.index(x))

data_2024['month_sin'] = np.sin(2*np.pi*data_2024['month_ID']/12)
data_2024['month_cos'] = np.cos(2*np.pi*data_2024['month_ID']/12)

model = make_pipeline(
    PolynomialFeatures(2),
    LinearRegression()
)
X = data_2024[['month_sin', 'month_cos']]
y = data_2024['Maximum_temperature']
model.fit(X, y)

jan_pred = model.predict([[np.sin(0), np.cos(0)]])[0]

actual_temp = 7.5

difference = jan_pred - actual_temp

print("=== Question 13 Results ===")
print(f"Predicted January 2025 temperature: {jan_pred:.2f}°C")
print(f"Difference (Predicted - Actual): {difference:.2f}°C")