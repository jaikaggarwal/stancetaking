from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.stats import binned_statistic_dd, truncnorm


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


def dbscan_clustering(data, epsilon):
    dbscan = DBSCAN(eps=epsilon)
    clusters = dbscan.fit(data)

    return clusters


def get_bins(data, num_bins):
    """Return the bin index that each data point in data falls into, given the space
    is subdivided to have num_bins equally sized bins.

    A bin number of i means that the corresponding value is between bin_edges[i-1], bin_edges[i]

    Returns both the bin index as a unique integer, as well as in terms of a 5d
    array corresponding to each dimension.
    """
    # Initialize uniformly-sized bins
    bin_edges = np.linspace(0, 1, (num_bins + 1))

    # TO DO: Can we modify the statistic to directly calculate a vector valued statistic?
    stats, edges, binnumber = binned_statistic_dd(data, np.arange(len(data)),
                                                  statistic="mean",
                                                  bins=[bin_edges for i in range(data.shape[1])])
    
    stats, edges, unraveled_binnumber = binned_statistic_dd(data, np.arange(len(data)),
                                                            statistic="mean",
                                                            bins=[bin_edges for i in range(data.shape[1])],
                                                            expand_binnumbers=True)

    # Return the bin IDs
    return binnumber, unraveled_binnumber.transpose()


def get_bin_centroids(data, bin_idx):
    """Calculate the centroid of all the points that lie within each bin.
    
    Use get_bins on the data first to get the bin_idx for each point.
    """
    bins = defaultdict(list)
    for point, bin in zip(data, bin_idx):
        bins[bin].append(point)

    for points in bins:
        bins[points] = np.mean(bins[points], axis=0)

    return bins


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

    dbscan_clusters = dbscan_clustering(data, 1)
    print("Silhouette score for DBSCAN:", silhouette_score(data, dbscan_clusters.labels_))

    
