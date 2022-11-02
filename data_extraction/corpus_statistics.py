import pandas as pd
import numpy as np
import os

from tqdm import tqdm


ROOT_DIR = "/ais/hal9000/datasets/reddit/stance_analysis/"
files = list(os.walk(ROOT_DIR))

total = None
for dir_tup in files:
    dir = dir_tup[0]
    if not dir.endswith("files"):
        continue
    if not (dir[-8:-6] in ["02", "05", "08", "11"]):
        continue
    sub_files = sorted(dir_tup[2])
    print(dir)
    for sub_file in tqdm(sub_files):
        df = pd.read_json(dir + "/" + sub_file, lines=True)
        tmp = df[df['BF'] == 1][['body', 'subreddit']]
        agg = tmp.groupby("subreddit").count()
        if total is None:
            total = agg
        else:
            total = total.add(agg, fill_value=0)
    total.to_csv("community_posting_statistics_2_5_8_11.csv")