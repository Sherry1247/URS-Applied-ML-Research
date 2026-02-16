import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score

path = "unsupervised_learning/archive/Iris.csv"
df = pd.read_csv(path).drop(columns=["Id"])

feature_cols = ["SepalLengthCm", "SepalWidthCm",
                "PetalLengthCm", "PetalWidthCm"]

X = df[feature_cols].values
y = df["Species"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to 2D for later experiments
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- K-Means on petal-only features ---
X_petal = df[["PetalLengthCm", "PetalWidthCm"]].values

scaler_petal = StandardScaler()
X_petal_scaled = scaler_petal.fit_transform(X_petal)

kmeans_petal = KMeans(n_clusters=3, random_state=42, n_init="auto")
clusters_petal = kmeans_petal.fit_predict(X_petal_scaled)

sil_petal = silhouette_score(X_petal_scaled, clusters_petal)
ari_petal = adjusted_rand_score(y, clusters_petal)

print("Petal-only KMeans -> silhouette:", sil_petal)
print("Petal-only KMeans -> ARI:", ari_petal)

# visualize in petal space
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_petal_scaled[:, 0],
                y=X_petal_scaled[:, 1],
                hue=clusters_petal.astype(str),
                palette="tab10")
plt.xlabel("PetalLength (scaled)")
plt.ylabel("PetalWidth (scaled)")
plt.title("K-Means on petal-only features")
plt.show()

# --- K-Means on 2D PCA representation ---
kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init="auto")
clusters_pca = kmeans_pca.fit_predict(X_pca)

sil_pca = silhouette_score(X_pca, clusters_pca)
ari_pca = adjusted_rand_score(y, clusters_pca)

print("PCA-space KMeans -> silhouette:", sil_pca)
print("PCA-space KMeans -> ARI:", ari_pca)

# visualize clusters in PCA space
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                hue=clusters_pca.astype(str), palette="tab10")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means clusters in PCA space (2D)")
plt.show()

# --- GMM on 2D PCA representation ---
gmm = GaussianMixture(n_components=3, covariance_type="full",
                      random_state=42)
gmm_labels = gmm.fit_predict(X_pca)  # hard labels from maximum probability

sil_gmm = silhouette_score(X_pca, gmm_labels)
ari_gmm = adjusted_rand_score(y, gmm_labels)

print("GMM (PCA space) -> silhouette:", sil_gmm)
print("GMM (PCA space) -> ARI:", ari_gmm)

# visualize GMM clusters in PCA space
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                hue=gmm_labels.astype(str), palette="tab10")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("GMM clusters in PCA space")
plt.show()

# --- Baseline: K-Means on all 4 standardized features ---
kmeans_all = KMeans(n_clusters=3, random_state=42, n_init="auto")
clusters_all = kmeans_all.fit_predict(X_scaled)

sil_all = silhouette_score(X_scaled, clusters_all)
ari_all = adjusted_rand_score(y, clusters_all)

print("All-feature KMeans -> silhouette:", sil_all)
print("All-feature KMeans -> ARI:", ari_all)


results = pd.DataFrame([
    ["KMeans_all_features",  sil_all,   ari_all],
    ["KMeans_petal_only",    sil_petal, ari_petal],
    ["KMeans_PCA_2D",        sil_pca,   ari_pca],
    ["GMM_PCA_2D",           sil_gmm,   ari_gmm],
], columns=["Method", "Silhouette", "ARI"])

print(results)

