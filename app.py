import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# Set Streamlit page config
st.set_page_config(page_title="Iris K-Means Clustering", layout="centered")

st.title("🌸 K-Means Clustering on Iris Dataset")

# Load dataset
iris = load_iris()
X = iris.data
y_true = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)

# Sidebar: K value
k = st.sidebar.slider("Select number of clusters (k)", min_value=1, max_value=10, value=3, step=1)

# Apply KMeans
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
df['Cluster'] = labels

# Optional: Visualize with PCA
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)

# Plotting
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(reduced_X[:, 0], reduced_X[:, 1], c=labels, cmap='viridis', s=50)
ax.set_title("K-Means Clustering (PCA-reduced Iris Data)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.grid(True)
st.pyplot(fig)

# Matching function
def match_labels(true_labels, cluster_labels):
    labels_matched = np.zeros_like(cluster_labels)
    for i in np.unique(cluster_labels):
        mask = (cluster_labels == i)
        most_common = mode(true_labels[mask], keepdims=True).mode[0]
        labels_matched[mask] = most_common
    return labels_matched

# Show clustering accuracy if k == 3
if k == 3:
    matched_labels = match_labels(y_true, labels)
    accuracy = accuracy_score(y_true, matched_labels)
    st.success(f"✅ Clustering Accuracy (after label alignment): **{accuracy:.2f}**")
else:
    st.warning("⚠️ Accuracy comparison requires k = 3 to match the true number of species.")

# Show DataFrame
if st.checkbox("Show clustered data table"):
    st.dataframe(df)
