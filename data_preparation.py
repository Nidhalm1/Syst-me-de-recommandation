import pandas as pd

def load_data(path):
    # Détection automatique du format MovieLens 100K
    if path.endswith("u.data"):
        df = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        df = df[['user_id', 'item_id', 'rating']]
        return df
    else:
        # Charger les données depuis un CSV en ignorant les lignes de commentaires
        return pd.read_csv(path, comment='#')

def preprocess_data(df):
    # Nettoyage basique : suppression des valeurs manquantes
    df = df.dropna(subset=['user_id', 'item_id', 'rating'])
    # Conversion des types
    df['user_id'] = df['user_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    df['rating'] = df['rating'].astype(float)
    return df

if __name__ == "__main__":
    # Utiliser le fichier d'exemple fourni
    df = load_data("exemple_user_ratings.csv")  # <-- Remplacé "u.data" par "exemple_user_ratings.csv"
    df = preprocess_data(df)
    df.to_csv("clean_user_ratings.csv", index=False)
