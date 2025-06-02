import pandas as pd

data = pd.read_csv('data1.csv')

unites = {
    'Minimum_temperature': '°C',
    'Maximum_temperature': '°C',
    'Rainfall': 'mm',
    'Sunshine_duration': 'heures'
}

variances = {
    "Variance de la température minimale": {
        'valeur': data['Minimum_temperature'].var(),
        'unité': unites['Minimum_temperature']
    },
    "Variance de la température maximale": {
        'valeur': data['Maximum_temperature'].var(),
        'unité': unites['Maximum_temperature']
    },
    "Variance de la hauteur de précipitations": {
        'valeur': data['Rainfall'].var(),
        'unité': unites['Rainfall']
    },
    "Variance de la durée d'ensoleillement": {
        'valeur': data['Sunshine_duration'].var(),
        'unité': unites['Sunshine_duration']
    }
}

print("ANALYSE DE VARIANCE AVEC UNITÉS")
print("="*45 + "\n")

for variable, result in variances.items():
    print(f"{variable}:")
    print(f"- Valeur: {result['valeur']:.2f} {result['unité']}²")
    print(f"- Interprétation: {'Variabilité élevée' if result['valeur'] > 100 else 'Variabilité modérée'}")
    print("\n" + "-"*45)

print("\nRÉCAPITULATIF (variance en unités carrées)")
print("-"*45)
df = pd.DataFrame.from_dict({
    k: [f"{v['valeur']:.2f} {v['unité']}²"] 
    for k, v in variances.items()
}, orient='index', columns=['Variance'])
print(df)