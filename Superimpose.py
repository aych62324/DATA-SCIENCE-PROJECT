import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data2.csv')

# Create month order for proper sorting
month_order = ['january', 'february', 'march', 'april', 'may', 'june', 
               'july', 'august', 'september', 'october', 'november', 'december']

# Filter data for each year
data_2023 = data[data['Year'] == 2023].copy()
data_2024 = data[data['Year'] == 2024].copy()

# Convert month names to numerical values for sorting
data_2023['month_num'] = data_2023['Month'].apply(lambda x: month_order.index(x))
data_2024['month_num'] = data_2024['Month'].apply(lambda x: month_order.index(x))

# Sort data by month
data_2023 = data_2023.sort_values('month_num')
data_2024 = data_2024.sort_values('month_num')

# Create the plot
plt.figure(figsize=(12, 6))

# Plot 2023 data in blue
plt.plot(data_2023['Month'], data_2023['Maximum_temperature'], 
         marker='o', color='blue', label='2023')

# Plot 2024 data in red
plt.plot(data_2024['Month'], data_2024['Maximum_temperature'], 
         marker='s', color='red', label='2024')

# Add labels and title
plt.title('Maximum Temperature Evolution in Paris (2023 vs 2024)')
plt.xlabel('Month')
plt.ylabel('Maximum Temperature (°C)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# Add data labels
for i, row in data_2023.iterrows():
    plt.text(row['Month'], row['Maximum_temperature']+0.5, 
             f"{row['Maximum_temperature']}°C", ha='center', color='blue')
    
for i, row in data_2024.iterrows():
    plt.text(row['Month'], row['Maximum_temperature']-0.5, 
             f"{row['Maximum_temperature']}°C", ha='center', color='red')

plt.tight_layout()
plt.show()