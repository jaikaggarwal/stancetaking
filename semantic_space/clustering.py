import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.stats import truncnorm


def generate_synthetic_data(samples):
    data = [truncnorm.rvs(0, 1, loc=0.5, scale=0.5) for _ in range(samples * 5)]
    data = np.array(data).reshape(samples, 5)
    return data


def svd(data):
    return np.linalg.svd(data)


def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters, random_state=42)
    clusters = kmeans.fit(data)
    
    return clusters


def dbscan_clustering(data):
    dbscan = DBSCAN(eps=1)
    clusters = dbscan.fit(data)

    return clusters


def plot_silhouette_scores(scores):
    plt.plot(np.arange(2, 11), scores, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Score by Number of K-Means Clusters")
    plt.savefig("figs/kmeans_silhouette_scores.png")


if __name__ == "__main__":
    data = generate_synthetic_data(10000)
    u, s, v = svd(data)

    print(u.shape)

    silhouette_scores = []
    # Find best number of kmeans clusters
    for n_clusters in range(2, 11):
        kmeans_clusters = kmeans_clustering(data, n_clusters)
        score = silhouette_score(data, kmeans_clusters.labels_)
        silhouette_scores.append(score)
        print(f"{n_clusters} clusters, silhouette score: {score}")

    plot_silhouette_scores(silhouette_scores)

    dbscan_clusters = dbscan_clustering(data)
    print("Silhouette score for DBSCAN:", silhouette_score(data, dbscan_clusters.labels_))

    
