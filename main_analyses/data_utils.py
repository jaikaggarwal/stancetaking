import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic_dd
import pandas as pd
from itertools import product
import os
import sys
from scipy.spatial.distance import pdist
from sklearn.manifold import MDS
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append("../data_extraction/")
from utils import flatten_logic, Serialization
from functools import reduce
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
tqdm.pandas()

ROOT_DIR = "/ais/hal9000/datasets/reddit/stance_pipeline/dec_6_full_run/full_features/"
FEATURE_COLUMNS = ['Valence', 'Arousal', 'Dominance', 'Politeness', 'Formality']
NUM_QUANTILES = 4
lemmatizer = WordNetLemmatizer()
crossposting_df = pd.read_csv("../data_extraction/crossposting_full_community_data.csv")


# Maps each feature to its quantile number in the semantic situation name (e.g. V1A2D1P2F1)
feature_to_index = {
    "Valence": 1,
    "Arousal": 3,
    "Dominance": 5,
    "Politeness": 7,
    "Formality": 9
}

quantile_to_plot_markers = {
    2: ["ro", "gs"],
    3: ["ro", "bx", "gs"],
    4: ["ro", "bo", "ys", "gs"]
}

quantile_to_legend_labels = {
    2: ["Low", "High"],
    3: ["Low", "Mid", "High"],
    4: ["Low", "Midlow", "Midhigh", "High"]
}



def get_bins(data, num_bins):
    """Return the bin index that each data point in data falls into, given the space
    is subdivided to have num_bins equally sized bins.

    A bin number of i means that the corresponding value is between bin_edges[i-1], bin_edges[i]

    Returns both the bin index as a unique integer, as well as in terms of a 5d
    array corresponding to each dimension.
    """
    # Initialize uniformly-sized bins
    bin_edges = []
    for feature in FEATURE_COLUMNS:
        bin_edges.append(np.quantile(data[feature], np.linspace(0, 1, num_bins + 1)))
    bin_edges = np.array(bin_edges)
    bin_edges[:, 0] = 0
    bin_edges[:, -1] = 1

    data = data.to_numpy()
    
    stats, edges, unraveled_binnumber = binned_statistic_dd(data, np.arange(len(data)),
                                                            statistic="mean",
                                                            bins=bin_edges,
                                                            expand_binnumbers=True)

    # Return the bin IDs
    return unraveled_binnumber.transpose()


def get_bin_names(arr):
    features = np.array(list("VADPF"))
    added = np.char.add(features, arr.astype(str))
    names = np.sum(added.astype(object), axis=1)
    return names


def load_data_from_raw():
    files = sorted(os.listdir(ROOT_DIR))
    dfs = []
    for file in tqdm(files):
        df = pd.read_csv(ROOT_DIR + file)
        dfs.append(df)
    df = pd.concat(dfs)
    del dfs
    df = df.set_index("id")
    print("Getting marker")
    df['rel_marker'] = df['rel_marker'].progress_apply(lambda x: eval(x)[0])
    print("Rescaling Politeness")
    df['Politeness'] = (df["Politeness"] - df['Politeness'].min())/(df['Politeness'].max() - df['Politeness'].min())

    print("Extracting Bins")
    ubins = get_bins(df[FEATURE_COLUMNS], NUM_QUANTILES)
    print("Getting bin names")
    df['bin'] = get_bin_names(ubins)
    print("Describing bin")
    df['bin'].describe()

    print("Getting mean data")
    x = df.groupby("bin").mean()[FEATURE_COLUMNS]
    print("Saving mean data")
    Serialization.save_obj(x, f"semantic_situation_mean_values_{NUM_QUANTILES}_full_data")
    print("Saving entire dataset")
    Serialization.save_obj(df[['subreddit', 'rel_marker', 'bin', 'Valence', 'Arousal', 'Dominance', 'Politeness', 'Formality']], f"stance_pipeline_full_data_{NUM_QUANTILES}_quantiles_full_data")

    return df, x


def get_sub_marker_pairs(df):
    all_markers = sorted(df['rel_marker'].unique())
    all_markers = [marker for marker in all_markers if marker not in ["'d", "10x"]]
    df = df[df['rel_marker'].isin(all_markers)]
    # Combine the subreddit and marker and aggregate
    df['sub_marker'] = df["subreddit"] + "_" + df['rel_marker']
    return df


def get_bin_com_markers(df):
    comms = df['subreddit'].unique()
    markers = df['rel_marker'].unique()
    bins = df['bin'].unique()
    com_markers = list(product(comms, markers))
    com_markers = ["_".join(pair) for pair in com_markers]
    return bins, comms, markers, com_markers


def get_need_probabilities(df, bins, comms):
    # Need probability
    # Takes 30 seconds to run
    # sem_sit_counts = df.groupby(["bin"]).count()['sub_marker']
    sem_sit_counts_per_community = df.groupby(["subreddit", "bin"]).count()[['sub_marker', "Valence"]]
    all_sub_counts = pd.DataFrame(0, index=pd.MultiIndex.from_product([bins, comms], names=["bin", "subreddit"]), columns=sem_sit_counts_per_community.columns)
    sem_sit_counts_per_community = sem_sit_counts_per_community.add(all_sub_counts, fill_value=0)
    sem_sit_counts_per_community['percent'] = sem_sit_counts_per_community.groupby(level=0)['sub_marker'].transform(lambda x: (x / x.sum()))
    # norm_sem_sit_counts = sem_sit_counts/sem_sit_counts.sum()
    # print(norm_sem_sit_counts.describe())
    com_to_need = {}
    for sub in comms:
        need_vec = sem_sit_counts_per_community.loc[sub]['percent']
        com_to_need[sub] = need_vec.to_numpy()
    need_df = pd.DataFrame(com_to_need).T
    need_df.columns = sem_sit_counts_per_community.loc[sub].index
    return com_to_need


def get_nonzero_prop(df):
    print(np.round(np.count_nonzero(df)/df.size, 2))


def lemmatize_markers(markers):
    markers = sorted(markers)
    lemmatized = []
    for token in tqdm(markers):
        if token.endswith("ing"):
            lemmatized.append(token)
        else:
            doc = lemmatizer.lemmatize(token, wordnet.VERB)
            lemmatized.append(doc)
    marker_to_lemma = dict(zip(markers, lemmatized))
    marker_to_adjusted_lemma = {}
    lemma_counts = Counter(lemmatized)
    for marker in markers:
        if lemma_counts[marker_to_lemma[marker]] > 1:
            marker_to_adjusted_lemma[marker] = marker_to_lemma[marker]
        else:
            marker_to_adjusted_lemma[marker] = marker

def pmi(df, positive=True):
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    df = df / expected
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
        df = np.nan_to_num(df)
    return df


def create_derived_representation(matrix, include_ppmi=True):
    matrix_np = matrix.to_numpy()
    print(matrix_np.shape)
    if include_ppmi:
        svd_input = pmi(matrix_np)
    else:
        svd_input = matrix_np
    get_nonzero_prop(svd_input)
    P, D, Q = np.linalg.svd(svd_input, full_matrices=False)
    output = {
        "svd_input": svd_input,
        "sem_rep": P,
        "singular_values": D,
        "marker_rep": Q,
        'sem_loadings': np.multiply(P, D),
        'marker_loadings': np.multiply(D.reshape(-1, 1), Q)
    }
    
    return output


def scree_plot(sing_val_matrix, to_show=True):
    eigenvalues = sing_val_matrix**2

    var_explained = sing_val_matrix**2/np.sum(sing_val_matrix**2)
    total_var_explained = np.concatenate(([0], np.cumsum(var_explained)))
    var_explained = np.round(var_explained, 3)
    if to_show:
        plt.clf()
        plt.plot(np.arange(len(eigenvalues[:10])) + 1, eigenvalues[:10])
        plt.xlabel("Component Number")
        plt.ylabel("Eigenvalue (x10^6)")
        plt.title("Eigenvalues of Each Component")
        plt.show()
        print(var_explained[:10])
        print(total_var_explained[:10])
        plt.clf()
        plt.plot(np.arange(len(var_explained[:10])) + 1, var_explained[:10])
        plt.xlabel("Component Number")
        plt.ylabel("Variance Explained")
        plt.title("Proportion of Variance Explained by Each Component")
        plt.show()
        plt.clf()
        plt.plot(np.arange(-1, len(total_var_explained[:10])) + 1, total_var_explained[:11])
        plt.ylim(0, np.max(total_var_explained[:11]) + 0.02)
        plt.xlabel("Component Number")
        plt.ylabel("Cumulative Variance Explained")
        plt.title("Cumulative Variance Explained Across Components")
        plt.show()
    return eigenvalues, var_explained, total_var_explained


def get_abs_diff(pair):
    bin_1, bin_2 = pair
    return np.abs(x.loc[bin_1].to_numpy() - x.loc[bin_2].to_numpy())

def get_diff(pair):
    bin_1, bin_2 = pair
    return x.loc[bin_1].to_numpy() - x.loc[bin_2].to_numpy()


def get_regression_matrix(bin_means, derived_space, abs_diff=True, to_save=False, save_name = ""):
    if abs_diff:
        diff_function = get_abs_diff
    else:
        diff_function = get_diff
    rows = []
    indices = []
    with Pool(6) as p:
        for i in tqdm(range(len(bin_means.index)-1)):
            bin_1 = bin_means.index.tolist()[i]
            input_vals = list(product([bin_1], bin_means.iloc[i+1:].index))
            r = list(p.imap(diff_function, input_vals))
            rows.extend(r)
            indices.extend(input_vals)
    sem_sit_sims_derived = pd.DataFrame(cosine_similarity(derived_space), columns=bin_means.index, index=bin_means.index)
    pairwise_distances = pd.DataFrame(np.array(rows), columns=FEATURE_COLUMNS, index=pd.MultiIndex.from_tuples(indices))
    pairwise_distances['Derived_Dist'] = pairwise_distances.progress_apply(lambda y: sem_sit_sims_derived.loc[y.name[0]][y.name[1]], axis=1)
    
    if to_save:
        Serialization.save_obj(pairwise_distances, save_name)


def calculate_pairwise_sim(data):
    pairwise_sim = 1 - pd.DataFrame(cosine_similarity(data, data), index=x.index, columns=x.index)
    pairwise_sim = pairwise_sim.where(np.triu(np.ones(pairwise_sim.shape), k=1).astype(np.bool))
    pairwise_sim = pairwise_sim.stack().rename("Original_Dist")
    pairwise_sim.index = pairwise_sim.index.rename(["Sit 1", "Sit 2"])
    return pd.DataFrame(pairwise_sim)


def mds_visualization_2d(lowdim, labels, feature, delegate, plot_markers, delegate_labels, legend_labels, derived_label, to_annotate):
    # Rewrite into wrapper function that computes mds just once and can be used for an arbitrary number of features
    plt.rcParams.update({"font.size": 20})
    fig = plt.figure(figsize=(20, 10))
    for i in range(len(delegate_labels)):
        vals = np.array([idx for idx, label in enumerate(labels) if delegate(label, delegate_labels[i], feature)]).astype(int)
        plt.plot(lowdim[vals, 0], lowdim[vals, 1], plot_markers[i], markersize=12, label=f"{legend_labels[i]} {feature}")
    plt.legend()
    if to_annotate:
        for i in range(len(lowdim)):
            plt.text(lowdim[i,0]*1.05, lowdim[i,1]*1.01, labels[i],fontsize=17)

    plt.xlabel('Dimension 1')    
    plt.ylabel('Dimension 2')
    plt.title(f"MDS Plot of {derived_label} Semantic Situation Space ({feature})")



def mds_visualization_3d(lowdim, labels, feature, delegate, plot_markers, delegate_labels, legend_labels, derived_label, to_annotate):
    # Rewrite into wrapper function that computes mds just once and can be used for an arbitrary number of features
    ax = plt.figure(figsize=(20, 10)).add_subplot(projection='3d')
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.5, 1.5, 1.5, 1]))
    # ax.set_box_aspect(aspect = (1,1,2))
    for i in range(len(delegate_labels)):
        vals = np.array([idx for idx, label in enumerate(labels) if delegate(label, delegate_labels[i], feature)]).astype(int)
        ax.plot(lowdim[vals, 0], lowdim[vals, 1], lowdim[vals, 2], plot_markers[i], markersize=8, label=f"{legend_labels[i]} {feature}")
    ax.legend(loc='upper right')
    if to_annotate:
        for i in range(len(lowdim)):
            ax.text(lowdim[i,0]*1.05, lowdim[i,1]*1.01, lowdim[i,2]*1.01, labels[i],fontsize=17)

    ax.set_xlabel('Dimension 1')    
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.view_init(0, -75)
    plt.title(f"MDS Plot of {derived_label} Semantic Situation Space ({feature})")



def mds_wrapper(data, labels, features, delegate, num_quantiles=2, num_components=2, is_derived=False, to_annotate=False):
    embedding = MDS(n_components=num_components, random_state=42)
    F_lowdim = embedding.fit_transform(data)
    print(F_lowdim.shape)

    plot_markers = quantile_to_plot_markers[num_quantiles]
    derived_label = "Derived" if is_derived else "Original"
    delegate_labels = [str(i+1) for i in range(num_quantiles)]
    legend_labels = quantile_to_legend_labels[num_quantiles]

    if num_components == 2:
        for feature in features:
            mds_visualization_2d(F_lowdim, labels, feature, delegate, plot_markers, delegate_labels, legend_labels, derived_label, to_annotate)
    else:
        for feature in features:
            mds_visualization_3d(F_lowdim, labels, feature, delegate, plot_markers, delegate_labels, legend_labels, derived_label, to_annotate)


