import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

# Load the data
data = pd.read_csv('online_shoppers_intention.csv')

# Select the columns for analysis
columns_for_analysis = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration']

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[columns_for_analysis])

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

# Combine the PCA results with the target variable "Revenue"
pca_df['Revenue'] = data['Revenue']

# Plot PCA visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Revenue', palette='Set1')
plt.title('PCA Visualization')
plt.show()

# Perform UMAP
umap_result = umap.UMAP(n_components=2).fit_transform(data_scaled)

# Create a DataFrame with the UMAP results
umap_df = pd.DataFrame(data=umap_result, columns=['UMAP1', 'UMAP2'])

# Combine the UMAP results with the target variable "Revenue"
umap_df['Revenue'] = data['Revenue']

# Plot UMAP visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Revenue', palette='Set1')
plt.title('UMAP Visualization')
plt.show()
