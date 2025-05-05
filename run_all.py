import subprocess

# 1. Préparation des données
print("Étape 1 : Préparation des données...")
try:
    subprocess.run(["python", "data_preparation.py"], check=True)
except subprocess.CalledProcessError as e:
    print("Erreur lors de la préparation des données :", e)
    exit(1)

# 2. Exploration des données
print("Étape 2 : Exploration des données...")
try:
    subprocess.run(["python", "exploration.py"], check=True)
except subprocess.CalledProcessError as e:
    print("Erreur lors de l'exploration des données :", e)
    exit(1)

# 3. Entraînement et recommandations
print("Étape 3 : Entraînement et recommandations...")
try:
    subprocess.run(["python", "recommender.py"], check=True)
except subprocess.CalledProcessError as e:
    print("Erreur lors de l'entraînement/recommandation :", e)
    exit(1)

# 4. Benchmarking
print("Étape 4 : Benchmarking...")
try:
    subprocess.run(["python", "benchmarking.py"], check=True)
except subprocess.CalledProcessError as e:
    print("Erreur lors du benchmarking :", e)
    exit(1)

# 5. Version Spark (optionnelle, nécessite Spark installé)
try:
    print("Étape 5 : Version Spark...")
    subprocess.run(["spark-submit", "spark_recommender.py"], check=True)
except Exception as e:
    print("Spark non disponible ou erreur lors de l'exécution Spark :", e)
