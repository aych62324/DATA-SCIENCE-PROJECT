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

plt.figure(figsize=(18, 14))

base_scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2, c='gray', s=40, label='Autres villes')

circle = plt.Circle((0,0), 1, color='blue', fill=False, linestyle='--', alpha=0.5)
plt.gca().add_artist(circle)

for i, var in enumerate(variables):
    plt.arrow(0, 0, 
              pca.components_[0, i]*0.85,
              pca.components_[1, i]*0.85,
              head_width=0.07, head_length=0.07, 
              fc='red', ec='red', alpha=0.8,
              length_includes_head=True,
              width=0.01,
              shape='full')
    plt.text(pca.components_[0, i]*0.95,
             pca.components_[1, i]*0.95, 
             var, color='darkred',
             ha='center', va='center',
             fontsize=12, weight='bold',
             bbox=dict(facecolor='white', alpha=0.7, pad=2, edgecolor='none'))

extremes = [
    {'var': 'Minimum_temperature', 'op': 'idxmin', 'label': 'Temp. min la plus basse', 'color': 'blue', 'marker': 'v'},
    {'var': 'Minimum_temperature', 'op': 'idxmax', 'label': 'Temp. min la plus haute', 'color': 'dodgerblue', 'marker': '^'},
    {'var': 'Maximum_temperature', 'op': 'idxmin', 'label': 'Temp. max la plus basse', 'color': 'darkred', 'marker': 'v'},
    {'var': 'Maximum_temperature', 'op': 'idxmax', 'label': 'Temp. max la plus haute', 'color': 'red', 'marker': '^'},
    {'var': 'Rainfall', 'op': 'idxmin', 'label': 'Précip. les plus faibles', 'color': 'limegreen', 'marker': 's'},
    {'var': 'Rainfall', 'op': 'idxmax', 'label': 'Précip. les plus fortes', 'color': 'darkgreen', 'marker': 's'},
    {'var': 'Sunshine_duration', 'op': 'idxmin', 'label': 'Ensoleil. le plus faible', 'color': 'gold', 'marker': 'D'},
    {'var': 'Sunshine_duration', 'op': 'idxmax', 'label': 'Ensoleil. le plus fort', 'color': 'orange', 'marker': 'D'}
]

city_manager = {}

for ex in extremes:
    idx = getattr(data[ex['var']], ex['op'])()
    city = data.loc[idx, 'City']
    x, y = X_pca[idx, 0], X_pca[idx, 1]
    
    if city in city_manager:
        angle = 2 * np.pi * city_manager[city]['count'] / 6
        offset = 0.2
        x += offset * np.cos(angle)
        y += offset * np.sin(angle)
        city_manager[city]['count'] += 1
    else:
        city_manager[city] = {'count': 1, 'original': (x, y)}
    
    point = plt.scatter(x, y,
               c=ex['color'], marker=ex['marker'],
               s=150, edgecolor='black',
               linewidth=1.2, alpha=0.9,
               label=ex['label'])

    if city_manager[city]['count'] > 1:
        orig_x, orig_y = city_manager[city]['original']
        plt.plot([orig_x, x], [orig_y, y], 'k:', alpha=0.4, lw=1)

    plt.text(x, y, city,
             fontsize=10, ha='center', va='bottom',
             bbox=dict(facecolor='white', alpha=0.8, pad=2, edgecolor='none'))

plt.axhline(0, color='grey', linestyle=':', alpha=0.3)
plt.axvline(0, color='grey', linestyle=':', alpha=0.3)
plt.xlabel(f"Composante Principale 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=13)
plt.ylabel(f"Composante Principale 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=13)
plt.title("Analyse ACP: Variables météo et villes extrêmes", fontsize=15, pad=20)
plt.grid(True, alpha=0.2)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
legend = plt.legend(by_label.values(), by_label.keys(),
           loc='upper left', bbox_to_anchor=(1.02, 1),
           title="Légende des extrêmes",
           frameon=True, framealpha=0.9,
           borderpad=1, handletextpad=1.5)
legend.get_title().set_fontsize(12)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("SYNTHÈSE DES VALEURS EXTRÊMES".center(80))
print("="*80)
units = {'Minimum_temperature': '°C', 'Maximum_temperature': '°C', 
         'Rainfall': 'mm', 'Sunshine_duration': 'heures'}
for ex in extremes:
    idx = getattr(data[ex['var']], ex['op'])()
    city = data.loc[idx, 'City']
    value = data.loc[idx, ex['var']]
    x, y = X_pca[idx, 0], X_pca[idx, 1]
    print(f"{ex['label'] + ':':<25} {city:<10} {value:>6.1f} {units[ex['var']]:<5} (Position: [{x:>5.2f}, {y:>5.2f}])")