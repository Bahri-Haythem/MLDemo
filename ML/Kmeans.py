import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


mean1 = [np.random.randint(50), np.random.randint(50)]
mean2 = [np.random.randint(50), np.random.randint(50)]
#Select two random points coordinates

cov = [[100,0], [0, 100]]

x1, y1 = np.random.multivariate_normal(mean1, cov, 100).T
x2, y2 = np.random.multivariate_normal(mean2, cov, 100).T
#Draw random samples from a multivariate normal distribution.

x = np.append(x1, x2)
y = np.append(y1, y2)

w = [[i, j] for i, j in zip(x,y)]

plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()


X = np.array(w)
print(X.tolist())

#select number of clusters
kmeans = KMeans(n_clusters = 2)
#train model
kmeans.fit(X)

#extract centroids
centroids = kmeans.cluster_centers_
#extract labels (in ur case 0 or 1)
labels = kmeans.labels_

print(labels)

colors = ["g.", "r."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=150, zorder=10)

plt.show()

#found by the model after training
print(centroids)
print(mean1, mean2)