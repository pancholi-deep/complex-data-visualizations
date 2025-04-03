import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('MovieReviews.csv')

# Preprocess text data (optional, based on the specific dataset and requirements)
# For example, remove punctuation, convert to lowercase, remove stop words, etc.

# Create the TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the number of features as needed
tfidf_matrix = tfidf_vectorizer.fit_transform(data['review'])

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

# Combine the PCA results with the target variable "sentiment"
pca_df['sentiment'] = data['sentiment']

# Plot PCA visualization
plt.figure(figsize=(8, 6))
plt.scatter(pca_df.loc[pca_df['sentiment']=='positive', 'PC1'], pca_df.loc[pca_df['sentiment']=='positive', 'PC2'], label='Positive', alpha=0.7)
plt.scatter(pca_df.loc[pca_df['sentiment']=='negative', 'PC1'], pca_df.loc[pca_df['sentiment']=='negative', 'PC2'], label='Negative', alpha=0.7)
plt.legend()
plt.title('PCA Visualization')
plt.show()

# Perform UMAP
umap_result = umap.UMAP(n_components=2).fit_transform(tfidf_matrix.toarray())

# Create a DataFrame with the UMAP results
umap_df = pd.DataFrame(data=umap_result, columns=['UMAP1', 'UMAP2'])

# Combine the UMAP results with the target variable "sentiment"
umap_df['sentiment'] = data['sentiment']

# Plot UMAP visualization
plt.figure(figsize=(8, 6))
plt.scatter(umap_df.loc[umap_df['sentiment']=='positive', 'UMAP1'], umap_df.loc[umap_df['sentiment']=='positive', 'UMAP2'], label='Positive', alpha=0.7)
plt.scatter(umap_df.loc[umap_df['sentiment']=='negative', 'UMAP1'], umap_df.loc[umap_df['sentiment']=='negative', 'UMAP2'], label='Negative', alpha=0.7)
plt.legend()
plt.title('UMAP Visualization')
plt.show()
