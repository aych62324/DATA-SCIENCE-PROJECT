import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline

data = pd.read_csv('data2.csv')
data_2024 = data[data['Year'] == 2024].copy()

month_order = ['january', 'february', 'march', 'april', 'may', 'june', 
               'july', 'august', 'september', 'october', 'november', 'december']
data_2024['month_ID'] = data_2024['Month'].apply(lambda x: month_order.index(x))

data_2024['month_sin'] = np.sin(2*np.pi*data_2024['month_ID']/12)
data_2024['month_cos'] = np.cos(2*np.pi*data_2024['month_ID']/12)

results = []

for degree in [1, 2, 3]:
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    
    X = data_2024[['month_sin', 'month_cos']]
    y = data_2024['Maximum_temperature']
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1)
    
    jan_pred = model.predict([[np.sin(0), np.cos(0)]])[0]
    
    results.append({
        'degree': degree,
        'r2': r2,
        'adj_r2': adj_r2,
        'prediction': jan_pred,
        'model': model
    })

best_model = max(results, key=lambda x: x['adj_r2'])

print("=== Cyclical Model Results ===")
print(f"Best polynomial degree: {best_model['degree']}")
print(f"R-squared: {best_model['r2']:.4f}")
print(f"Adjusted R-squared: {best_model['adj_r2']:.4f}")
print(f"Predicted Jan 2025 temp: {best_model['prediction']:.2f}°C")

months = np.linspace(0, 11, 100)
X_plot = pd.DataFrame({
    'month_sin': np.sin(2*np.pi*months/12),
    'month_cos': np.cos(2*np.pi*months/12)
})
y_plot = best_model['model'].predict(X_plot)

plt.figure(figsize=(12, 6))
plt.scatter(data_2024['month_ID'], data_2024['Maximum_temperature'], 
            color='black', label='2024 Data')
plt.plot(months, y_plot, color='red', label='Model Prediction')
plt.scatter([0], [best_model['prediction']], color='blue', 
            s=100, label='Jan 2025 Prediction')
plt.axhline(y=7.5, color='green', linestyle='--', label='Actual Jan 2025')

plt.title('Temperature Prediction with Cyclical Encoding')
plt.xlabel('Month ID (0=Jan)')
plt.ylabel('Temperature (°C)')
plt.xticks(range(12), month_order, rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

error = best_model['prediction'] - 7.5
print(f"\n=== Error Analysis ===")
print(f"Absolute error: {abs(error):.2f}°C")
print(f"Percentage error: {abs(error)/7.5*100:.1f}%")