import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("clean_user_ratings.csv")

# Statistiques descriptives
print(df.describe())

# Distribution des notes
plt.hist(df['rating'], bins=10)
plt.title("Distribution des notes")
plt.xlabel("Note")
plt.ylabel("Nombre")
plt.show()
