import pandas as pd
from recommender import create_matrix, compute_svd_centered, recommend

def test_recommend():
    # Données factices
    df = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "item_id": [10, 20, 10, 30],
        "rating": [5, 3, 4, 2]
    })
    matrix = create_matrix(df)
    U, sigma, Vt, user_means = compute_svd_centered(matrix, k=2)
    items, scores = recommend(1, matrix, U, sigma, Vt, user_means, n=2)
    assert len(items) == 2
    print("Test recommend OK")

if __name__ == "__main__":
    test_recommend()
