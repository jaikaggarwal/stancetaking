import argparse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from nltk import word_tokenize, sent_tokenize
from utils import Serialization
from feature_extraction import LexicalAnalysis
from extract_embeddings_utils import SBERT

tqdm.pandas()

FEATURES = ["valence", "arousal", "dominance", "formality", "politeness"]
LENGTHS = [4, 6, 8, 10, 12, 14, 16] #TODO: You can adjust these to ones that seem sensible (mb 4, 6, 8, 10, 16)



def extract_relevant_markers(new_line, terms):
    """Collect all of the relevant markers that are present
    within new_line."""
    curr_body = set(new_line.lower().split(" "))
    present_markers = [val for val in terms if val in curr_body]
    return present_markers


def get_space_tokenize_length(text: str):
    return len(text.split())

def get_word_tokenize_length(text: str):
    return len(nltk.word_tokenize(text))

def get_tokenization_length_method(tokenization_method: str):
    """Return a tokenization function based on the name.
    
    Accepted methods are:
        - "spaces" which tokenizes a sentence by spaces.
        - "word" which uses nltk.word_tokenize.
        - "SBERT" which uses the embedded SBERT tokenizer.
    """
    if tokenization_method == "spaces":
        tokenizer_function = get_space_tokenize_length
    elif tokenization_method == "word":
        tokenizer_function = get_word_tokenize_length
    elif tokenization_method == "SBERT":
        sbert_model = SBERT()
        tokenizer_function == sbert_model.get_tokenize_lengths
    else:
        raise ValueError("No such tokenization method.")

    return tokenizer_function

# Gather sample data
def load_sample_data(tokenization_method: str):
    """
    Loads the first 500K comments from the 2014 Reddit data dumps (specifically, the first 500K
    comments of length greater than 4 according to word_tokenize).
    """
    # Create the relevant output folder
    if not os.path.exists("length_analysis/"):
        os.makedirs("length_analysis/")

    # Ensure that the path below is correct
    stance_groups = pd.read_json("../stancemarkers/stancemarkers.json").T
    # We retain only positive affect, negative affect, and emphatic stance markers
    sub_group = stance_groups[stance_groups['stance_group'].isin(["positive_affect_verbs", "positive_affect_adjective", "negative_affect_verbs", "negative_affect_adjective", "positive_affect_adverb", "negative_affect_adverb", "emphatic"])]
    # Keep only the relevant markers, and create a mapping from each marker to its group
    rel_markers = set(sub_group.index)
    marker_to_group = sub_group['stance_group'].to_dict()

    print("are we here?")
    # We can limit our analysis to the first 500K comments of 2014
    ROOT_DIR = "/ais/hal9000/datasets/reddit/stance_analysis/"
    files = sorted(list(os.walk(ROOT_DIR)))
    df = pd.read_json(files[1][0] + "/aa", lines=True)

    print("did we get here?")

    # For now, we can further limit our analysis to those with Biber and Finnegan markers
    df = df[df['BF'] == 1]
    # We need to split each post up into sentences, since that is the level of context we are working with
    df['sens'] = df['body'].progress_apply(lambda x: sent_tokenize(x))
    df_big = df.explode("sens")

    # Keep a subset of the columns for ease of storage
    tmp = df_big[['author', 'subreddit', 'id', 'sens']].reset_index(drop=True)
    # Keep only those sentences with exactly one relevant marker
    tmp['rel_marker'] = tmp['sens'].progress_apply(lambda x: extract_relevant_markers(x, rel_markers))
    tmp['one_marker'] = tmp['rel_marker'].apply(lambda x: len(x) == 1)
    tmp = tmp[tmp['one_marker']]
    tmp['marker_category'] = tmp['rel_marker'].apply(lambda x: marker_to_group[x[0]])
    
    # Here is where we calculate the length of sentences

    # TODO: Change lambda function to include tokenization_method
    tokenizer_function = get_tokenization_method(tokenization_method)
    tmp['len'] = tmp['sens'].progress_apply(lambda x: tokenizer_function(x))
    tmp = tmp.rename(columns={"sens": "body"})

    for i in LENGTHS:
        curr = tmp[tmp['len'] == i]
        curr['body_mask'] = curr.apply(lambda x: re.sub(x['rel_marker'][0], "[MASK]", x['body'].lower()), axis=1)
        curr.to_csv(f"length_analysis/{i}_len_posts_{tokenization_method}.csv")


def compute_feature_values(tokenization_method):
    valence_model = Serialization.load_obj('valence_model')
    arousal_model = Serialization.load_obj('arousal_model')
    dominance_model = Serialization.load_obj('dominance_model')
    politeness_model = Serialization.load_obj("wikipedia_politeness_classifier")
    formality_model = pickle.load(open("/ais/hal9000/datasets/reddit/stance_pipeline/classifiers/formality_model.sav", "rb"))
    sbert_model = SBERT()
    
    for i in LENGTHS:
        df = pd.read_csv(f"length_analysis/{i}_len_posts_{tokenization_method}.csv", index_col=0)
        masked_posts = df['body_mask'].tolist()
        embeddings = sbert_model.get_embeddings(masked_posts)
    
        vad_vals = LexicalAnalysis.infer_emotion_value(embeddings, valence_model, arousal_model, dominance_model)
        vad_vals = pd.DataFrame(np.asarray(vad_vals).T, index=m_file.index, columns=['valence', 'arousal', 'dominance'])
        df = pd.concat((df, vad_vals), axis=1)
        df['politeness'] = LexicalAnalysis.infer_politeness(embeddings, politeness_model)
        df['formality'] = LexicalAnalysis.infer_formality(embeddings, formality_model)
        df.to_csv(f"length_analysis/{i}_len_posts_{tokenization_method}_vadpf.csv", index_col=0)

    


def process_feature_data(tokenization_method):
    """
    Wrapper function for creating the plots. 
    """    
    feature_len_value_map = {feature: {} for feature in FEATURES}
    # First, we can load all the data into feature_len_value_map
    for i in LENGTHS:
        curr = pd.read_csv(f"length_analysis/{i}_len_posts_{tokenization_method}_vadpf.csv", index_col=0)
        for feature in FEATURES:
            feature_len_value_map[feature][i] = curr[f'{feature}_masked']

    # Now, we can iterate through each feature, and for each length value, plot the distribution
    # The variable base_length is used to plot each of the distributions against one we think is stable (length of 16)
    base_length = max(LENGTHS)
    for feature in FEATURES:
        for i in LENGTHS:
            plot_feature_distribution(feature, feature_len_value_map[feature], i, base_length)
    


def plot_feature_distribution(feature, len_to_feature, length, base_length):
    for key in [base_length, length]:
        print(f"Post Length: {key} Mean Valence: {len_to_feature[key].mean()} +- {len_to_feature[key].std()}")
        plt.hist(len_to_feature[key], density=True, alpha=1 if key == base_length else 0.4, label=f"{key}")
    plt.title(f"{feature.capitalize()} Score Distributions by Sentence Length")
    plt.xlabel(f"{feature.capitalize()} Scores")
    plt.ylabel("Probability Density")
    plt.legend()
    
    plt.savefig(f"feature_figures/{tokenization_method}_length_analysis_figs.png")
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run length analysis")
    parser.add_argument("tokenization_method", type=str, help="Tokenization method")
    parser.add_argument("--gpu_id", type=int, help="Number of cores to use.", default=3)
    
    
    args = parser.parse_args()

    # Set the gpu number appropriately
    from sentence_transformers.SentenceTransformer import torch as pt
    pt.cuda.set_device(args.gpu_id)
    print(pt.cuda.is_available())


    load_sample_data(args.tokenization_method)
    compute_feature_values(tokenization_method)
    process_feature_data()