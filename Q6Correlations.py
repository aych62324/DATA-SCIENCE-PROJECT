import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

data = pd.read_csv('data1.csv')

variables = ['Minimum_temperature', 'Maximum_temperature', 'Rainfall', 'Sunshine_duration']
data_meteo = data[variables + ['City']]

corr_matrix = data_meteo[variables].corr()

correlations = []
for i in range(len(variables)):
    for j in range(i+1, len(variables)):
        var1 = variables[i]
        var2 = variables[j]
        r_value = corr_matrix.loc[var1, var2]
        correlations.append((var1, var2, r_value))

corr_sorted = sorted(correlations, key=lambda x: abs(x[2]), reverse=True)

most_positive = max(correlations, key=lambda x: x[2])
most_negative = min(correlations, key=lambda x: x[2])
least_correlated = min(correlations, key=lambda x: abs(x[2]))

print("Résultats des corrélations:")
print("="*50)
print(f"1. Variables les plus positivement corrélées: {most_positive[0]} et {most_positive[1]}")
print(f"   Coefficient de corrélation: {most_positive[2]:.3f}\n")

print(f"2. Variables les plus négativement corrélées: {most_negative[0]} et {most_negative[1]}")
print(f"   Coefficient de corrélation: {most_negative[2]:.3f}\n")

print(f"3. Variables les moins corrélées: {least_correlated[0]} et {least_correlated[1]}")
print(f"   Coefficient de corrélation: {least_correlated[2]:.3f}")

def plot_correlation(var1, var2, title, r_value):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data_meteo, x=var1, y=var2, color='blue', alpha=0.7)

    for i in range(len(data_meteo)):
        plt.text(data_meteo[var1].iloc[i], 
                 data_meteo[var2].iloc[i], 
                 data_meteo['City'].iloc[i],
                 fontsize=8, ha='center', va='bottom')

    sns.regplot(data=data_meteo, x=var1, y=var2, 
               scatter=False, color='red', ci=None)
    
    plt.title(f"{title}\nCoefficient de corrélation: {r_value:.3f}")

    units = {'Minimum_temperature': '°C', 
             'Maximum_temperature': '°C',
             'Rainfall': 'mm',
             'Sunshine_duration': 'heures'}
    plt.xlabel(f"{var1} ({units[var1]})")
    plt.ylabel(f"{var2} ({units[var2]})")
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\nVisualisations:")
print("="*50)
plot_correlation(most_positive[0], most_positive[1], 
                "Corrélation positive la plus forte", most_positive[2])

plot_correlation(most_negative[0], most_negative[1], 
                "Corrélation négative la plus forte", most_negative[2])

plot_correlation(least_correlated[0], least_correlated[1], 
                "Corrélation la plus faible", least_correlated[2])