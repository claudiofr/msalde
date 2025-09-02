from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)

ft = pca.fit_transform([[1, 2], [3, 4], [5, 6]])
print(pca.components_)