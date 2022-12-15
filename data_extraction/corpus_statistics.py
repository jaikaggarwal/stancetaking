import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
import tqdm
import re
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from utils import Serialization

model_name="bert-large-nli-mean-tokens"
model = SentenceTransformer(model_name)


ROOT_DIR = "/ais/hal9000/datasets/reddit/stance_analysis/"
files = sorted(list(os.walk(ROOT_DIR)))

# stance_groups = pd.read_json("~/stancetaking/stancemarkers/stancemarkers.json").T
# sub_group = FIX#stance_groups[stance_groups['stance_group'].isin(["positive_affect_verbs", "positive_affect_adjective", "negative_affect_verbs", "negative_affect_adjective", "positive_affect_adverb", "negative_affect_adverb", "emphatic"])]
rel_markers = Serialization.load_obj("expanded_stancemarkers_0.85_threshold")
rel_markers.remove("is")
print(f"Number of markers: {len(rel_markers)}")
# marker_to_group = sub_group['stance_group'].to_dict()

def extract_relevant_markers(new_line, terms):
    curr_body = set(new_line.lower().split(" "))
    present_markers = [val for val in terms if val in curr_body]
    return present_markers

def get_sbert_length(sens):
    tokens = model.tokenize(sens)
    lens = [len(token_set) for token_set in tokens]
    return lens

def filter_df(df):
    df['sens'] = df['body'].apply(lambda x: sent_tokenize(x))
    df_big = df.explode("sens")
    tmp = df_big[['author', 'subreddit', "created_utc", 'id', 'sens']].reset_index(drop=True)
    tmp['rel_marker'] = tmp['sens'].apply(lambda x: extract_relevant_markers(x, rel_markers))
    tmp['one_marker'] = tmp['rel_marker'].apply(lambda x: len(x) == 1)
    tmp = tmp[tmp['one_marker']]
    # tmp['marker_category'] = tmp['rel_marker'].apply(lambda x: marker_to_group[x[0]])
    tmp['len'] = get_sbert_length(tmp['sens'].tolist())
    tmp = tmp.rename(columns={"sens": "body"})
    return tmp[tmp['len'] >= 6] # TODO

def extract_test_data(test_communities_file, data_dir, output_dir):
    print(data_dir)
    rel_communities = pd.read_csv(test_communities_file, index_col=0).index.tolist()
    print(f"Number of relevant communities: {len(rel_communities)}")
    sub_files = os.listdir(data_dir)
    curr_total = []
    for sub_file in sub_files:
        df = pd.read_json(data_dir + "/" + sub_file, lines=True)
        tmp = df[df['BF'] == 1][['author', 'body', 'subreddit', 'id', "created_utc", "BF_markers"]]
        tmp['subreddit'] = tmp['subreddit'].str.lower()
        tmp = tmp[tmp['subreddit'].isin(rel_communities)]
        curr_total.append(filter_df(tmp))
    idx = data_dir.rfind("/")
    agg = pd.concat(curr_total)
    # Mask sentences
    agg['body_mask'] = agg.apply(lambda x: re.sub(x['rel_marker'][0], "[MASK]", x['body'].lower()), axis=1)
    agg.to_csv(f"{output_dir}/{data_dir[idx + 1: ]}.csv")


def mask_sentences():
    files = sorted(os.listdir(ROOT_DIR + 'test_run_data/'))
    for file  in files:
        df = pd.read_csv(ROOT_DIR + "test_run_data/" + file, index_col=0).reset_index(drop=True)
        df['body_mask'] = df.apply(lambda x: re.sub(eval(x['rel_marker'])[0], "[MASK]", x['body'].lower()), axis=1)
        df.to_csv(ROOT_DIR + "test_run_data/" + file)



if __name__ == '__main__':
    dirs = [dir_tup[0] for dir_tup in files if dir_tup[0].endswith("files")]
    dirs = sorted(dirs)
    with Pool(6) as p:
        r = list(tqdm.tqdm(p.imap(extract_test_data, dirs), total=len(dirs)))



# total = None
# for dir_tup in files:
#     dir = dir_tup[0]
#     if not dir.endswith("files"):
#         continue
#     sub_files = sorted(dir_tup[2])
#     print(dir)
#     for sub_file in tqdm(sub_files):
#         df = pd.read_json(dir + "/" + sub_file, lines=True)
#         tmp = df[df['BF'] == 1][['body', 'subreddit']]
#         agg = tmp.groupby("subreddit").count()
#         if total is None:
#             total = agg
#         else:
#             total = total.add(agg, fill_value=0)
#     total.to_csv("community_posting_statistics_2_5_8_11.csv")