import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load
path = "unsupervised_learning/archive/Iris.csv"
df = pd.read_csv(path)
df = df.drop(columns=["Id"])

feature_cols = ["SepalLengthCm", "SepalWidthCm",
                "PetalLengthCm", "PetalWidthCm"]

X = df[feature_cols].values
y = df["Species"].values

# 2. Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === PCA: choose components ===
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

exp_var = pca_full.explained_variance_ratio_
cum_exp_var = np.cumsum(exp_var)

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(exp_var) + 1), cum_exp_var, marker="o")
plt.xticks(range(1, len(exp_var) + 1))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA cumulative explained variance")
plt.grid(True)
plt.show()

print("Explained variance ratio:", exp_var)
print("Cumulative explained variance:", cum_exp_var)

# === PCA to 2 components ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Species": y
})

plt.figure(figsize=(6, 5))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Species")
plt.title("Iris in PCA 2D space (colored by species)")
plt.show()

# === K-Means clustering on original scaled features ===
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X_scaled)

print("Silhouette score:", silhouette_score(X_scaled, clusters))
print("ARI vs true species:", adjusted_rand_score(y, clusters))

cluster_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": clusters.astype(str),  # as string for nicer legend
    "Species": y
})

plt.figure(figsize=(6, 5))
sns.scatterplot(data=cluster_df, x="PC1", y="PC2",
                hue="Cluster", palette="tab10")
plt.title("K-Means clusters in PCA space")
plt.show()

# Compare clusters vs true labels
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(ax=axes[0], data=cluster_df, x="PC1", y="PC2",
                hue="Species")
axes[0].set_title("True species")

sns.scatterplot(ax=axes[1], data=cluster_df, x="PC1", y="PC2",
                hue="Cluster", palette="tab10")
axes[1].set_title("K-Means clusters")
plt.tight_layout()
plt.show()

# === Classification on full 4D features ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === Classification on 2D PCA features + decision regions ===
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

clf_pca = LogisticRegression(max_iter=200)
clf_pca.fit(X_pca_train, y_pca_train)

# Grid over PC1–PC2 space
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z_labels = clf_pca.predict(grid)  # string labels

# Encode labels to integers for contourf
le = LabelEncoder()
le.fit(y)                      # fit on all species
Z_int = le.transform(Z_labels) # numeric
Z_int = Z_int.reshape(xx.shape)

plt.figure(figsize=(6, 5))
# decision regions
plt.contourf(xx, yy, Z_int, alpha=0.3, levels=len(le.classes_))
# data points
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y)
plt.title("Logistic regression decision regions in PCA space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
