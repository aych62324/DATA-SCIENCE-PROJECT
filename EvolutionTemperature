import pandas as pd
import matplotlib.pyplot as plt

data2 = pd.read_csv('data2.csv')

data_2024 = data2[data2['Year'] == 2024]

month_order = ['january', 'february', 'march', 'april', 'may', 'june', 
               'july', 'august', 'september', 'october', 'november', 'december']
data_2024['Month'] = pd.Categorical(data_2024['Month'], categories=month_order, ordered=True)
data_2024 = data_2024.sort_values('Month')

plt.figure(figsize=(10, 6))
plt.plot(data_2024['Month'], data_2024['Maximum_temperature'], marker='o', color='b')
plt.title('Evolution of Maximum Temperature in Paris (2024)')
plt.xlabel('Month')
plt.ylabel('Maximum Temperature (°C)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

for i, temp in enumerate(data_2024['Maximum_temperature']):
    plt.text(i, temp+0.5, f'{temp}°C', ha='center')

plt.show()

print("Maximum Temperatures in Paris (2024):")
print(data_2024[['Month', 'Maximum_temperature']].reset_index(drop=True))