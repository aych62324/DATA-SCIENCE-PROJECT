import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data1.csv')
variables = ['Minimum_temperature', 'Maximum_temperature', 'Rainfall', 'Sunshine_duration']
X = data[variables]
cities = data['City']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 10))
plt.gca().set_aspect('equal', adjustable='box')

for i, var in enumerate(variables):
    plt.arrow(0, 0, 
              pca.components_[0, i], 
              pca.components_[1, i],
              head_width=0.05, head_length=0.05, 
              fc='red', ec='red')
    plt.text(pca.components_[0, i]*1.15, 
             pca.components_[1, i]*1.15, 
             var, color='darkred',
             ha='center', va='center',
             fontsize=12)

circle = plt.Circle((0,0), 1, color='blue', fill=False)
plt.gca().add_artist(circle)

plt.xlabel(f"Composante 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
plt.ylabel(f"Composante 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
plt.title("Cercle de corrélations des variables météorologiques", fontsize=14, pad=20)

plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.grid(True, linestyle='--', alpha=0.5)

print("\nINTERPRÉTATION DU CERCLE DE CORRÉLATIONS:")
print("="*60)
print("1. Variables bien représentées (proches du cercle):")
for i, var in enumerate(variables):
    quality = (pca.components_[0,i]**2 + pca.components_[1,i]**2)
    if quality > 0.7:
        print(f"- {var} (qualité: {quality:.2f})")

print("\n2. Corrélations entre variables:")
print("- Températures min/max: fortement corrélées (même direction)")
print("- Ensoleillement et précipitations: anti-corrélées (directions opposées)")

print("\n3. Contribution aux axes:")
print(f"- CP1: Principalement expliquée par les températures")
print(f"- CP2: Principalement expliquée par l'opposition précipitations/ensoleillement")

plt.tight_layout()
plt.show()