import pandas as pd

path = "unsupervised_learning/archive/Iris.csv"
df = pd.read_csv(path)

print(df.head())
print(df.info())
print(df['Species'].value_counts())
print(df.describe())

##############Visualization################
import seaborn as sns
import matplotlib.pyplot as plt

# Drop Id column
df = df.drop(columns=['Id'])

# Pairplot colored by species
sns.pairplot(df, hue="Species", diag_kind="hist")
plt.suptitle("Iris pairplot by species", y=1.02)
plt.show()

# Feature distributions
feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

df[feature_cols].hist(figsize=(8, 6), bins=10)
plt.suptitle("Feature distributions", y=1.02)
plt.tight_layout()
plt.show()

# Correlation heatmap
corr = df[feature_cols].corr()
plt.figure(figsize=(5, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature correlation heatmap")
plt.show()

