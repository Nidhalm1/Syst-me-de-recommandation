import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split

def create_matrix(df):
    """
    Crée la matrice utilisateur-item, en pivotant et en remplissant les absences par 0.
    """
    required = {'user_id', 'item_id', 'rating'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")
    mat = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    return mat

def compute_biases(matrix, reg=10.0):
    """
    Calcule global_mean, biais utilisateur et biais item avec régularisation.
    reg = coefficient de régularisation (lambda).
    """
    # Construire la matrice avec NaN pour les zéros
    R = matrix.replace(0, np.nan)
    mu = R.stack().mean()
    # nombre de notes par user/item
    n_u = R.count(axis=1)
    n_i = R.count(axis=0)
    # biais régularisés
    b_u = ((R.sub(mu, axis=0)).sum(axis=1)) / (reg + n_u)
    b_i = ((R.sub(mu, axis=1)).sum(axis=0)) / (reg + n_i)
    return mu, b_u.fillna(0), b_i.fillna(0)

def compute_svd_biased(matrix, k=3, reg=10.0):
    """
    SVD sur la matrice centrée par biais (global, user, item).
    """
    mu, b_u, b_i = compute_biases(matrix, reg=reg)
    # Centre la matrice
    R_bias = matrix.copy().astype(float)
    for u in R_bias.index:
        for i in R_bias.columns:
            if R_bias.at[u, i] != 0:
                R_bias.at[u, i] -= (mu + b_u[u] + b_i[i])
    R = R_bias.values
    # Ajuste k
    min_dim = min(R.shape)
    k = min(k, min_dim - 1) if min_dim > 1 else 1
    # Décomposition
    U, s, Vt = svds(R, k=k)
    idx = np.argsort(s)[::-1]
    s, U, Vt = s[idx], U[:, idx], Vt[idx, :]
    sigma = np.diag(s)
    return mu, b_u, b_i, U, sigma, Vt

def predict_rating(u_idx, i_idx, mu, b_u, b_i, U, sigma, Vt):
    """
    Prédit la note centrée plus biais.
    """
    pred = mu + b_u.iloc[u_idx] + b_i.iloc[i_idx] + U[u_idx, :].dot(sigma).dot(Vt[:, i_idx])
    # Clip entre 1 et 5
    return np.clip(pred, 1.0, 5.0)

def recommend(user_id, matrix, mu, b_u, b_i, U, sigma, Vt, n=5):
    """
    Recommande les n items non évalués les mieux classés pour un utilisateur.
    """
    if user_id not in matrix.index:
        raise ValueError(f"Utilisateur {user_id} inconnu")
    u_idx = matrix.index.get_loc(user_id)
    all_preds = []
    for i_idx, item_id in enumerate(matrix.columns):
        if matrix.iat[u_idx, i_idx] == 0:
            all_preds.append((item_id, predict_rating(u_idx, i_idx, mu, b_u, b_i, U, sigma, Vt)))
    all_preds.sort(key=lambda x: x[1], reverse=True)
    return all_preds[:n]

def evaluate(test_df, matrix, mu, b_u, b_i, U, sigma, Vt):
    """
    Calcule RMSE et MAE sur un jeu de test.
    """
    preds, trues = [], []
    idx_map = {u: idx for idx, u in enumerate(matrix.index)}
    col_map = {i: idx for idx, i in enumerate(matrix.columns)}
    for _, row in test_df.iterrows():
        u, i, true = row['user_id'], row['item_id'], row['rating']
        if u in idx_map and i in col_map:
            pred = predict_rating(idx_map[u], col_map[i], mu, b_u, b_i, U, sigma, Vt)
            preds.append(pred)
            trues.append(true)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    return rmse, mae

def cross_validate_svd(df, k_list=[2, 3, 4], reg_list=[10.0], folds=5):
    """
    Recherche du meilleur couple (k, reg) via cross-validation.
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    results = []
    for k in k_list:
        for reg in reg_list:
            rmses = []
            maes = []
            for train_idx, test_idx in kf.split(df):
                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                mat = create_matrix(train_df)
                mu, b_u, b_i, U, sig, Vt = compute_svd_biased(mat, k=k, reg=reg)
                rmse, mae = evaluate(test_df, mat, mu, b_u, b_i, U, sig, Vt)
                rmses.append(rmse)
                maes.append(mae)
            results.append({'k': k, 'reg': reg, 'rmse': np.mean(rmses), 'mae': np.mean(maes)})
    return pd.DataFrame(results)

if __name__ == '__main__':
    df = pd.read_csv('clean_user_ratings.csv')
    # Tune k et reg
    grid = cross_validate_svd(
        df,
        k_list=[2, 3, 4, 5, 6, 8, 10],
        reg_list=[1, 5, 10, 20],
        folds=3
    )
    print("Résultats CV pour différents (k, reg):")
    print(grid)
    best = grid.loc[grid['rmse'].idxmin()]
    best_k, best_reg = int(best.k), best.reg
    print(f"Meilleur nombre de facteurs latents: {best_k}, régularisation: {best_reg}")

    # Entraîner avec le meilleur k/reg
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    mat = create_matrix(train)
    mu, b_u, b_i, U, sig, Vt = compute_svd_biased(mat, k=best_k, reg=best_reg)
    rmse, mae = evaluate(test, mat, mu, b_u, b_i, U, sig, Vt)
    print(f"Test RMSE={rmse:.4f}, MAE={mae:.4f}")

    # Baseline
    from sklearn.metrics import mean_squared_error
    baseline_rmse = np.sqrt(mean_squared_error(test['rating'], [train['rating'].mean()] * len(test)))
    print(f"RMSE baseline (moyenne) : {baseline_rmse:.4f}")

    # Exemples de recommandations
    user = train['user_id'].iloc[0]
    print(f"Top recommandations pour l'utilisateur {user}:")
    print(recommend(user, mat, mu, b_u, b_i, U, sig, Vt, n=5))

    # Optionnel : afficher la matrice de corrélation utilisateur-item
    # print("Matrice de corrélation utilisateur-item :")
    # print(mat.corr())
