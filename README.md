# Système de recommandation personnalisé

Ce projet propose un système de recommandation de films basé sur les préférences utilisateurs, avec une version classique (Python/Scikit-learn) et une version distribuée (Spark MLlib).

## Structure du projet

- `data_preparation.py` : Préparation et nettoyage des données utilisateurs/films.
- `exploration.py` : Analyse exploratoire des données.
- `recommender.py` : Entraînement du modèle de recommandation et génération de recommandations.
- `benchmarking.py` : Évaluation et comparaison des performances des modèles.
- `spark_recommender.py` : Version distribuée utilisant Apache Spark.
- `run_all.py` : Script d’exécution automatique de toutes les étapes du projet.

## Prérequis

- Python 3.8+
- pip (gestionnaire de paquets Python)
- (Optionnel) Apache Spark pour la version distribuée

### Bibliothèques Python nécessaires

Installez les dépendances avec :

```bash
pip install -r requirements.txt
```

Exemple de contenu pour `requirements.txt` :
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```
Ajoutez `pyspark` si vous souhaitez utiliser Spark localement.

## Utilisation

Pour exécuter toutes les étapes du projet :

```bash
python run_all.py
```

Chaque étape s’exécutera dans l’ordre :
1. Préparation des données
2. Exploration des données
3. Entraînement et recommandations
4. Benchmarking
5. (Optionnel) Version Spark

Pour exécuter une étape spécifique, lancez le script correspondant :

```bash
python data_preparation.py
python exploration.py
python recommender.py
python benchmarking.py
spark-submit spark_recommender.py
```


## Auteurs

- Nidhal Moussa

## Licence

