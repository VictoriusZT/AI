import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

# Завантаження
X = np.loadtxt('data_clustering.txt', delimiter=',')

# Оцінка ширини вікна для Х
bandwidth_X = estimate_bandwidth(X, quantile=0.2, n_samples=len(X))

# Кластеризація даних методом зсуву середнього
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Витягування центрів кластерів
cluster_centers = meanshift_model.cluster_centers_
print("Centers of cluesters:\n", cluster_centers)

# Оцінка кількості кластерів
labels = meanshift_model.labels_
num_cluesters= len(np.unique(labels))
print("\nNumber of clusters in input data=", num_cluesters)

# Відображення на графіку точок та центрів кластерів
plt.figure()
markers = cycle('o*sv')
for i, marker in zip(range(num_cluesters), markers):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker, color='black')
# Відображення на графіку центру кластера
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o', markerfacecolor='black', markeredgecolor='black', markersize=15)
plt.title('Кластери')
plt.show()
