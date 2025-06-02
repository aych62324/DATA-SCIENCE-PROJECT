import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data1.csv')
X = data[['Minimum_temperature', 'Maximum_temperature', 'Rainfall', 'Sunshine_duration']]
cities = data['City']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

variance_ratio = pca.explained_variance_ratio_
pc1_var = variance_ratio[0] * 100
pc2_var = variance_ratio[1] * 100

plt.figure(figsize=(12, 10))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)

for i, city in enumerate(cities):
    plt.text(X_pca[i, 0], X_pca[i, 1], city, 
             fontsize=9, ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.3))

plt.xlabel(f'Composante Principale 1 ({pc1_var:.1f}% de variance)', fontsize=12)
plt.ylabel(f'Composante Principale 2 ({pc2_var:.1f}% de variance)', fontsize=12)
plt.title('Projection des villes sur les deux premières composantes principales', fontsize=14)

plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
plt.axvline(0, color='grey', linestyle='--', alpha=0.5)

print("\n" + "="*60)
print("VARIANCE EXPLIQUÉE PAR LES COMPOSANTES PRINCIPALES")
print("="*60)
print(f"Composante Principale 1 (CP1): {pc1_var:.1f}%")
print(f"Composante Principale 2 (CP2): {pc2_var:.1f}%")
print(f"Variance cumulée CP1+CP2: {pc1_var + pc2_var:.1f}%")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()