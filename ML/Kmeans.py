import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

mean1 = [np.random.randint(50), np.random.randint(50)]
mean2 = [np.random.randint(50), np.random.randint(50)]

cov = [[100,0], [0, 100]]

x1, y1 = np.random.multivariate_normal(mean1, cov, 100).T
x2, y2 = np.random.multivariate_normal(mean2, cov, 100).T

x = np.append(x1, x2)
y = np.append(y1, y2)

w = [[i, j] for i, j in zip(x,y)]

plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()

X = np.array(w)
print(X.tolist())
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

colors = ["g.", "r."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=150, zorder=10)

plt.show()

print(centroids)
print(mean1, mean2)