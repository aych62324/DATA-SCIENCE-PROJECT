import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('data1.csv')
variables = ['Minimum_temperature', 'Maximum_temperature', 'Rainfall', 'Sunshine_duration']
data_std = (data[variables] - data[variables].mean()) / data[variables].std()
data_std['City'] = data['City']

corr_ville = data_std.set_index('City').T.corr()

def get_unique_pairs(corr_matrix):
    corr = corr_matrix.stack()
    corr.index = corr.index.map(lambda x: tuple(sorted(x)))
    return corr[corr.index.map(lambda x: x[0] != x[1])].drop_duplicates()

unique_corr = get_unique_pairs(corr_ville)

print("3 paires de villes les plus similaires:")
print(unique_corr.sort_values(ascending=False).head(3))
print("\n3 paires de villes les plus opposées:")
print(unique_corr.sort_values().head(3))

plt.figure(figsize=(15, 12))
mask = np.triu(np.ones_like(corr_ville, dtype=bool))
sns.heatmap(corr_ville, mask=mask, cmap='coolwarm', center=0,
            annot=True, fmt=".2f", linewidths=0.5,
            vmin=-1, vmax=1, cbar_kws={'label': 'Coefficient de corrélation'})
plt.title('Matrice de corrélation entre villes (profil météorologique standardisé)\n', pad=20)
plt.xticks(rotation=90, fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()