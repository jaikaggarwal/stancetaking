import argparse
# import emoji
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
from scipy import stats
import sys
import time
from tqdm import tqdm
# tqdm.pandas()

DATA_OUT_DIR = '/ais/hal9000/datasets/reddit/stance_pipeline/classifiers/'
HTTP_PATTERN = re.compile(r'[\(\[]?https?:\/\/.*?(\s|\Z)[\r\n]*')

class Serialization:
    @staticmethod
    def save_obj(obj, name):
        """
        serialization of an object
        :param obj: object to serialize
        :param name: file name to store the object
        """
        with open(DATA_OUT_DIR + name + '.pkl', 'wb') as fout:
            pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)
        # end with
    # end def

    @staticmethod
    def load_obj(name):
        """
        de-serialization of an object
        :param name: file name to load the object from
        """
        with open(DATA_OUT_DIR + name + '.pkl', 'rb') as fout:
            return pickle.load(fout)


def set_intersection(l1, l2):
    """
    Returns the intersection of two lists.
    """
    return list(set(l1).intersection(set(l2)))


def set_union(l1, l2):
    """
    Returns the union of two lists.
    """
    return list(set(l1).union(set(l2)))

def set_difference(l1, l2):
    """
    Returns the difference of two lists.
    """
    return list(set(l1).difference(set(l2)))

def intersect_overlap(l1, l2):
    """
    Returns the intersection of two lists,
    while also describing the size of each list
    and the size of their intersection.
    """
    print(len(l1))
    print(len(l2))
    intersected = set_intersection(l1, l2)
    print(len(intersected))
    return intersected

def jaccard_similarity(l1, l2):
    l1 = set(l1)
    l2 = set(l2)
    return np.round(len(l1.intersection(l2)) / len(l1.union(l2)), 2)

def flatten_logic(arr):
    """
    Flattens a nested array. 

    """
    for i in arr:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def groupby_threshold(df, group_column, threshold_column, cutoff):
    """
    Return number of datapoints before and after thresholding.
    """
    agg = df.groupby(group_column).count()
    met_cutoff = agg[agg[threshold_column] >= cutoff].index.tolist()
    new_df = df[df[group_column].isin(met_cutoff)]
    print(f"Data size before applying threshold to {threshold_column}: {df.shape[0]}")
    print(f"Data size after applying threshold to {threshold_column}: {new_df.shape[0]}")
    return new_df


def flatten(arr):
    """
    Wrapper for the generator returned by flatten logic.
    """
    return list(flatten_logic(arr))


def preprocess(text):
    """
    Preprocesses text from Reddit posts and comments.
    """
    # Replace links with LINK token
    line = HTTP_PATTERN.sub(" LINK ", text)
    # Replace irregular symbol with whitespace
    line = re.sub("&amp;#x200b", " ", line)
    # Replace instances of users quoting previous posts with empty string
    line = re.sub(r"&gt;.*?(\n|\s\s|\Z)", " ", line)
    # Replace extraneous parentheses with whitespace
    line = re.sub(r'\s\(\s', " ", line)
    line = re.sub(r'\s\)\s', " ", line)
    # Replace newlines with whitespace
    line = re.sub(r"\r", " ", line)
    line = re.sub(r"\n", " ", line)
    # Replace mentions of users with USER tokens
    line = re.sub("\s/?u/[a-zA-Z0-9-_]*(\s|\Z)", " USER ", line)
    # Replace mentions of subreddits with REDDIT tokens
    line = re.sub("\s/?r/[a-zA-Z0-9-_]*(\s|\Z)", " REDDIT ", line)
    # Replace malformed quotation marks and apostrophes
    line = re.sub("’", "'", line)
    line = re.sub("”", '"', line)
    line = re.sub("“", '"', line)
    # Get rid of asterisks indicating bolded or italicized comments
    line = re.sub("\*{1,}(.+?)\*{1,}", r"\1", line)    
    # Replace emojis with EMOJI token
    # line = emoji.get_emoji_regexp().sub(" EMOJI ", line)
    # Replace all multi-whitespace characters with a single space.
    line = re.sub("\s{2,}", " ", line)
    return line


def find_controls(treatment, control, NUM_QUANTILES):
    sub = treatment.groupby("speaker").count()
    control_sub = control.groupby("speaker").count()
    sub_size = sub.shape[0]//(NUM_QUANTILES-1)
    q = [int(np.quantile(sub['text'], quantile)) for quantile in np.linspace(0, 1, NUM_QUANTILES)]
    dfs = []
    take = 0
    for i in range(0, NUM_QUANTILES-1):
        take += sub_size
        min_q = control_sub[control_sub['text'] >= q[i]]
        bound_q = min_q[min_q['text'] < q[i+1]]
        if len(bound_q) < take:
            take = take - len(bound_q) + 1
            dfs.append(bound_q)
        else:
            dfs.append(bound_q.sample(take))
            take = 0
    matched_control = pd.concat(dfs)
    print(len(sub))
    print(len(matched_control))
    print(sub.sum()['text'])
    print(matched_control.sum()['text'])
    return matched_control.index.tolist()

