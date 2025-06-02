import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data2.csv')

month_order = ['january', 'february', 'march', 'april', 'may', 'june', 
               'july', 'august', 'september', 'october', 'november', 'december']

data_2023 = data[data['Year'] == 2023].copy()
data_2024 = data[data['Year'] == 2024].copy()

data_2023['month_num'] = data_2023['Month'].apply(lambda x: month_order.index(x))
data_2024['month_num'] = data_2024['Month'].apply(lambda x: month_order.index(x))

data_2023 = data_2023.sort_values('month_num')
data_2024 = data_2024.sort_values('month_num')

plt.figure(figsize=(12, 6))

plt.plot(data_2023['Month'], data_2023['Maximum_temperature'], 
         marker='o', color='blue', label='2023')

plt.plot(data_2024['Month'], data_2024['Maximum_temperature'], 
         marker='s', color='red', label='2024')

plt.title('Maximum Temperature Evolution in Paris (2023 vs 2024)')
plt.xlabel('Month')
plt.ylabel('Maximum Temperature (°C)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

for i, row in data_2023.iterrows():
    plt.text(row['Month'], row['Maximum_temperature']+0.5, 
             f"{row['Maximum_temperature']}°C", ha='center', color='blue')
    
for i, row in data_2024.iterrows():
    plt.text(row['Month'], row['Maximum_temperature']-0.5, 
             f"{row['Maximum_temperature']}°C", ha='center', color='red')

plt.tight_layout()
plt.show()