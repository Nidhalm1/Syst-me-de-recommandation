from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

spark = SparkSession.builder.appName("Recommender").getOrCreate()
df = spark.read.csv("clean_user_ratings.csv", header=True, inferSchema=True)

# Split train/test
train, test = df.randomSplit([0.8, 0.2], seed=42)

als = ALS(userCol="user_id", itemCol="item_id", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(train)

predictions = model.transform(test)
predictions.show(5)
