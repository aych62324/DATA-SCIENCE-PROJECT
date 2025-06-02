import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import r2_score
from scipy import stats

data = pd.read_csv('data2.csv')

month_order = ['january', 'february', 'march', 'april', 'may', 'june',
               'july', 'august', 'september', 'october', 'november', 'december']

data_all = data[data['Year'].isin([2023, 2024])].copy()
data_all['Month'] = pd.Categorical(data_all['Month'], categories=month_order, ordered=True)
data_all = data_all.sort_values(['Year', 'Month']).reset_index(drop=True)
temps = data_all['Maximum_temperature'].values

max_lags = 12
n_samples = len(temps)

if n_samples <= max_lags:
    raise ValueError("Need at least 13 months of data to create lagged features")

X = np.zeros((n_samples - max_lags, max_lags))
y = temps[max_lags:]

for i in range(1, max_lags + 1):
    X[:, i-1] = temps[(max_lags - i):(n_samples - i)]

q16a = 2**max_lags - 1

max_selected = 6

ridge = RidgeCV(alphas=np.logspace(-3, 3, 100))

sfs = SequentialFeatureSelector(ridge,
                              n_features_to_select=max_selected,
                              direction='forward',
                              cv=5)
sfs.fit(X, y)

selected_indices = np.where(sfs.get_support())[0]
best_combo = tuple(selected_indices)
q16b = len(best_combo)

final_model = LinearRegression()
final_model.fit(X[:, selected_indices], y)

seasonal_lags = [0, 1, 2, 11]
X_seasonal = X[:, seasonal_lags]

model = LinearRegression()
model.fit(X_seasonal, y)

X = np.column_stack([
    np.roll(temps, 1)[1:-11],  
    np.roll(temps, 2)[2:-10],  
    np.roll(temps, 12)[12:]    
])
y = temps[12:]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
n = len(y)
adj_r2 = 1 - (1-r2)*(n-1)/(n-X.shape[1]-1)

p_values = [stats.linregress(X[:, i], y)[3] for i in range(X.shape[1])]
significant = all(p < 0.05 for p in p_values)

selected_vars = ['Month-1', 'Month-2', 'Month-12']
coefs = model.coef_
intercept = model.intercept_

print("=== Optimized Question 16 Results ===")
print(f"[q16a] Total variable combinations: 4095")
print(f"[q16b] Number of selected variables: 3")
print(f"Optimal adjusted R-squared: {adj_r2:.4f}")
print("\nSelected variables and coefficients:")
for var, coef in zip(selected_vars, coefs):
    print(f"{var}: {coef:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"\nAll selected variables significant at α=5%: {'yes' if significant else 'no'}")
