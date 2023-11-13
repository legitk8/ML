from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=500, n_features=2, centers=3)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
km.fit(X)

centroids = km.cluster_centers_

colors_dict = {0: 'red', 1: 'orange', 2: 'green'}

import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], color=[colors_dict[label] for label in km.labels_])
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='black')
plt.show()

# elbow method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)
    
plt.plot(range(1,11), wcss)
plt.show()