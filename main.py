import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("customers.csv")

print("Dataset Preview:")
print(df.head())

# Features
X = df

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

# Optional clustering (important for portfolio)
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Plot
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels)
plt.title("Customer Segmentation using UMAP")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
