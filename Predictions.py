import pandas as pd
import numpy as np

data = pd.read_csv('data2.csv')
month_order = ['january', 'february', 'march', 'april', 'may', 'june',
               'july', 'august', 'september', 'october', 'november', 'december']

data_all = data.copy()
data_all['month_num'] = data_all['Month'].apply(lambda x: month_order.index(x))
data_all = data_all.sort_values(['Year', 'month_num'])

dec_2024_temp = data_all[(data_all['Year'] == 2024) & (data_all['Month'] == 'december')]['Maximum_temperature'].values[0]
nov_2024_temp = data_all[(data_all['Year'] == 2024) & (data_all['Month'] == 'november')]['Maximum_temperature'].values[0]
jan_2024_temp = data_all[(data_all['Year'] == 2023) & (data_all['Month'] == 'january')]['Maximum_temperature'].values[0]

intercept = 3.2158
coef_month1 = 0.4271
coef_month2 = 0.1983
coef_month12 = 0.3526


pred_jan_2025 = intercept + \
                coef_month1 * dec_2024_temp + \
                coef_month2 * nov_2024_temp + \
                coef_month12 * jan_2024_temp

q17 = pred_jan_2025 - 7.5

print("=== Question 17 Results ===")
print(f"Predicted January 2025 temperature: {pred_jan_2025:.2f}°C")
print(f"[q17] Difference (Predicted - Actual): {q17:.2f}°C")

print("\nModel Evaluation for Subsequent Months:")
print("Note: These predictions use previous predicted values as inputs")
print("Month    | Predicted | Actual | Difference")
print("-----------------------------------------")

feb_2024_temp = data_all[(data_all['Year'] == 2023) & (data_all['Month'] == 'february')]['Maximum_temperature'].values[0]
pred_feb_2025 = intercept + \
                coef_month1 * pred_jan_2025 + \
                coef_month2 * dec_2024_temp + \
                coef_month12 * feb_2024_temp
print(f"February | {pred_feb_2025:.2f}°C  | 8.6°C  | {pred_feb_2025-8.6:.2f}°C")

mar_2024_temp = data_all[(data_all['Year'] == 2023) & (data_all['Month'] == 'march')]['Maximum_temperature'].values[0]
pred_mar_2025 = intercept + \
                coef_month1 * pred_feb_2025 + \
                coef_month2 * pred_jan_2025 + \
                coef_month12 * mar_2024_temp
print(f"March    | {pred_mar_2025:.2f}°C  | 14.6°C | {pred_mar_2025-14.6:.2f}°C")

apr_2024_temp = data_all[(data_all['Year'] == 2023) & (data_all['Month'] == 'april')]['Maximum_temperature'].values[0]
pred_apr_2025 = intercept + \
                coef_month1 * pred_mar_2025 + \
                coef_month2 * pred_feb_2025 + \
                coef_month12 * apr_2024_temp
print(f"April    | {pred_apr_2025:.2f}°C  | 20.0°C | {pred_apr_2025-20.0:.2f}°C")