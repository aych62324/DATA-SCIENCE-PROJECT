import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

data = pd.read_csv('data2.csv')
data_2024 = data[data['Year'] == 2024].copy()

month_order = ['january', 'february', 'march', 'april', 'may', 'june', 
               'july', 'august', 'september', 'october', 'november', 'december']
data_2024['month_ID'] = data_2024['Month'].apply(lambda x: month_order.index(x))

X = data_2024['month_ID'].values.reshape(-1, 1)
y = data_2024['Maximum_temperature'].values

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

p_value = model.pvalues[1]  # [q14a]

significant = "yes" if p_value < 0.05 else "no"

print("=== Question 14 Results ===")
print(f"p-value for β1: {p_value:.6f}")
print(f"Significant linear relationship (α=5%): {significant}")