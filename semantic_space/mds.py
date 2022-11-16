"Calculate and visualize similarity between clusters using MDS"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import truncnorm
from scipy.spatial.distance import pdist
from sklearn.manifold import MDS
from clustering import generate_synthetic_data

np.random.seed(42)

def generate_synthetic_data(samples, features):
    data = [truncnorm.rvs(0, 1, loc=0.5, scale=0.5) for _ in range(samples * features)]
    data = np.array(data).reshape(samples, features)
    return data


def get_distance_matrix(data, metric=None):
    """Return a distance matrix that can be used to calculate distance
    between clusters, using a supplied metric.

    By default calculate cosine distance.
    """
    if metric is None:
        return pdist(data, metric="cosine")
    else:
        return pdist(data, metric=metric)


def visualize_mds(data, labels, title, filename):
    """Data should be a sense count x feature matrix where each row corresponds to the centroid
    of a sense.
    
    Labels should be the same as length data"""
    assert len(data) == len(labels)

    embedding = MDS(n_components=2)
    F_lowdim = embedding.fit_transform(data)

    fig = plt.figure(figsize=(20, 10))
    plt.plot(F_lowdim[:, 0], F_lowdim[:, 1], "bo", markersize=12)

    for i in range(len(F_lowdim)):
        plt.text(F_lowdim[i,0]*1.05, F_lowdim[i,1]*1.01, labels[i],fontsize=17)
    
    plt.xlabel('Dimension 1')    
    plt.ylabel('Dimension 2')
    plt.title(title)

    plt.savefig(filename)


if __name__ == "__main__":
    # --- Uncomment the below for sample MDS on synthetic data ---
    # sense_counts = 8
    # k = 12
    # og_centroids = generate_synthetic_data(sense_counts, 5)
    # derived_centroids = generate_synthetic_data(sense_counts, k)
    # sense_labels = [f"sense{i}" for i in range(8)]
    # visualize_mds(og_centroids, sense_labels, "Original space MDS", "figs/original_mds.png")
    # visualize_mds(derived_centroids, sense_labels, "Derived space MDS", "figs/derived_mds.png")

    data_path = "/ais/hal9000/datasets/reddit/jai_stance_embeddings/unmasked/output/original_space_data.csv"
    data = pd.read_csv(data_path)

    sense_labels = data["bin"]
    values = data.loc[:, data.columns != "bin"]
    visualize_mds(values, sense_labels, "Original space MDS", "figs/reddit_original_mds.png")
