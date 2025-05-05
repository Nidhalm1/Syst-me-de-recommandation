import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from recommender import create_matrix, compute_svd_biased, cross_validate_svd
import numpy as np
from sklearn.model_selection import train_test_split

def evaluate(test, matrix, mu, b_u, b_i, U, sigma, Vt):
    preds = []
    trues = []
    idx_map = {u: idx for idx, u in enumerate(matrix.index)}
    col_map = {i: idx for idx, i in enumerate(matrix.columns)}
    for _, row in test.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        if user_id in idx_map and item_id in col_map:
            user_idx = idx_map[user_id]
            item_idx = col_map[item_id]
            pred = mu + b_u.iloc[user_idx] + b_i.iloc[item_idx] + U[user_idx, :].dot(sigma).dot(Vt[:, item_idx])
            preds.append(pred)
            trues.append(row['rating'])
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mse)
    return rmse, mae

if __name__ == "__main__":
    df = pd.read_csv("clean_user_ratings.csv")
    # Recherche du meilleur (k, reg) via CV
    print("Recherche du meilleur (k, reg) via cross-validation...")
    grid = cross_validate_svd(
        df,
        k_list=[2, 3, 4, 5, 6, 8, 10],
        reg_list=[1, 5, 10, 20],
        folds=3
    )
    print(grid)
    best = grid.loc[grid['rmse'].idxmin()]
    best_k, best_reg = int(best.k), best.reg
    print(f"Meilleur k trouvé : {best_k}, régularisation : {best_reg}")

    # Split train/test
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    matrix = create_matrix(train)
    mu, b_u, b_i, U, sigma, Vt = compute_svd_biased(matrix, k=best_k, reg=best_reg)
    rmse, mae = evaluate(test, matrix, mu, b_u, b_i, U, sigma, Vt)
    print(f"Test RMSE={rmse:.4f}, MAE={mae:.4f}")

    # Baseline
    baseline_mse = mean_squared_error(test['rating'], [train['rating'].mean()] * len(test))
    baseline_rmse = np.sqrt(baseline_mse)
    print(f"RMSE baseline (moyenne) : {baseline_rmse:.4f}")

    # Gain
    gain = baseline_rmse - rmse
    print(f"Gain (baisse du RMSE) : {gain:.4f}")
