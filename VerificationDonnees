import pandas as pd

def analyser_donnees_meteo(fichier='data1.csv'):

    try:
        df = pd.read_csv(fichier)
    except FileNotFoundError:
        print(f"Erreur: Le fichier {fichier} n'a pas été trouvé dans le dossier courant.")
        return None

    nombre_villes = df.shape[0]
    print(f"Nombre de villes dans le dataset : {nombre_villes}")

    donnees_manquantes = df.isnull().sum()
    villes_avec_manques = df[df.isnull().any(axis=1)]
    
    print("\nVérification des données manquantes :")
    print(donnees_manquantes)
    
    if villes_avec_manques.empty:
        print("\nAucune ville n'a de données manquantes.")
        nb_villes_manquantes = 0
    else:
        nb_villes_manquantes = len(villes_avec_manques)
        print(f"\nNombre de villes avec des données manquantes : {nb_villes_manquantes}")
        print("\nDétail des villes avec données manquantes :")
        print(villes_avec_manques[['City'] + list(donnees_manquantes[donnees_manquantes > 0].index)])
    
    resultats = {
        'Question': ['q1a', 'q1b'],
        'Réponse': [nombre_villes, nb_villes_manquantes]
    }
    

if __name__ == "__main__":
    analyser_donnees_meteo()