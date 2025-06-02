import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data1.csv')

variances = {
    'Minimum_temperature': data['Minimum_temperature'].var(),
    'Maximum_temperature': data['Maximum_temperature'].var(), 
    'Rainfall': data['Rainfall'].var(),
    'Sunshine_duration': data['Sunshine_duration'].var()
}

var_min = min(variances, key=variances.get)
var_max = max(variances, key=variances.get)

display_names = {
    'Minimum_temperature': 'Température minimale',
    'Maximum_temperature': 'Température maximale',
    'Rainfall': 'Précipitations',
    'Sunshine_duration': 'Durée d\'ensoleillement'
}

units = {
    'Minimum_temperature': '°C',
    'Maximum_temperature': '°C',
    'Rainfall': 'mm',
    'Sunshine_duration': 'heures'
}

def analyze_variable(var_key):
    values = data[var_key]
    display_name = display_names[var_key]
    unit = units[var_key]

    stats = {
        'Moyenne': f"{values.mean():.2f} {unit}",
        'Médiane': f"{values.median():.2f} {unit}",
        'Ecart-type': f"{values.std():.2f} {unit}"
    }

    plt.figure(figsize=(14, 8))
    ax = plt.gca()

    n, bins, patches = plt.hist(values, bins=8, color='#4e79a7',
                               edgecolor='white', alpha=0.9,
                               rwidth=0.8)
    
    print(f"\nVérification pour {display_name}:")
    counts = []
    for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
        in_bin = ((values >= bin_start) & (values <= bin_end)).sum()
        counts.append(in_bin)
        print(f"Barre {i+1}: {bin_start:.1f}-{bin_end:.1f} {unit} -> {in_bin} villes")
    
    y_offsets = {i: 0.5 for i in range(len(bins)-1)}
    
    for _, row in data.iterrows():
        value = row[var_key]
        city = row['City']
        bin_idx = np.digitize(value, bins) - 1
        bin_idx = max(0, min(len(bins)-2, bin_idx))
        x_pos = (bins[bin_idx] + bins[bin_idx+1]) / 2
        y_pos = y_offsets[bin_idx]
        
        ax.text(x_pos, y_pos, city,
               ha='center', va='bottom',
               fontsize=9, color='white',
               bbox=dict(facecolor='#4e79a7', alpha=0.7, pad=0.3))
        
        y_offsets[bin_idx] += 1
 
    plt.title(f'Distribution de {display_name}\n', fontsize=14, pad=20)
    plt.xlabel(f'{display_name} ({unit})', fontsize=12)
    plt.ylabel('Nombre de villes', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    stats_text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    plt.show()
    
    return stats, counts

print("\n" + "="*60)
print("VARIABLE AVEC VARIANCE MINIMALE")
stats_min, counts_min = analyze_variable(var_min)

print("\n" + "="*60)
print("VARIABLE AVEC VARIANCE MAXIMALE")
stats_max, counts_max = analyze_variable(var_max)

print("\n" + "="*60)
print(f"Total villes analysées - Températures: {sum(counts_min)}")
print(f"Total villes analysées - Ensoleillement: {sum(counts_max)}")