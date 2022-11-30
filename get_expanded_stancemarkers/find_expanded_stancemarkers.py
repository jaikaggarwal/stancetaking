import argparse
from collections import defaultdict
import json

import numpy as np
from gensim.models import KeyedVectors
from gensim import models

from tqdm import tqdm
from collections import Counter
from nltk import word_tokenize
from multiprocessing import Pool

import sys
sys.path.append("../data_extraction/")
from utils import Serialization, flatten

def load_wang2vec_model(model_path):
    return models.KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')

def load_stancemarkers(stance_path):
    with open(stance_path) as stancemarkers:
        markers = json.load(stancemarkers)
        return markers


def find_similar_stancemarkers(w2v_model, stancemarkers, out_path, cutoff=0.75):
    """Return a dictionary of words that have cosine similarities above some cutoff
    with respect to any of a list of stancemarkers.
    
    The keys are the word and the value is a list of all stancemarkers they are similar
    to.
    """
    expanded_lexicon = {}
    keys = np.asarray(w2v_model.index_to_key)
    valid_stance_markers = set(Serialization.load_obj("valid_stance_markers"))
    for stancemarker in tqdm(stancemarkers):
        if stancemarker in keys:
            # Find all the word embeddings from wang2vec with cosine similarity
            # greater than cutoff with this stancemarker
            sims = w2v_model.cosine_similarities(w2v_model[stancemarker], w2v_model.vectors)
            similar_markers = sims >= cutoff
            val_keys = keys[similar_markers]
            val_sims = sims[similar_markers]

            expanded_lexicon[stancemarker] = {}
            expanded_lexicon[stancemarker]['group'] = stancemarkers[stancemarker]["stance_group"]
            expanded_lexicon[stancemarker]['marker_sims'] = {}
            for similar_marker, sim_value in zip(val_keys, val_sims):
                if similar_marker in valid_stance_markers:
                    expanded_lexicon[stancemarker]["marker_sims"][similar_marker] = sim_value
                    # expanded_lexicon[similar_marker].add(stancemarkers[stancemarker]["stance_group"])
    Serialization.save_obj(expanded_lexicon, "stance_expanded_lexicon")
    # json_object = json.dumps(expanded_lexicon)
 
    # # Writing to sample.json
    # with open(out_path, "w") as outfile:
    #     outfile.write(json_object)
    # return expanded_lexicon


def expanded_stancemarkers_to_json(expanded_stancemarkers, out_path):
    """Convert dictionary of expanded stancemarkers to json file in the same format
    as original stancemarkers.json.
    """
    expanded_lexicon = {}
    for marker in expanded_stancemarkers:
        expanded_lexicon[marker] = {
            "source": "wang2vec",
            "stance_group": list(expanded_stancemarkers[marker])
        }

    json_object = json.dumps(expanded_lexicon)
 
    # Writing to sample.json
    with open(out_path, "w") as outfile:
        outfile.write(json_object)


def get_word_by_frequency():
    word_freqs = "/ais/hal9000/datasets/reddit/stance_analysis/wang2vec_sample/reddit_dataset.txt"
    big_collection = Counter()
    counter = 0
    batch_size = 500000
    with open(word_freqs, "r") as file:
        batch = []
        for line in tqdm(file, total=21750005):
            counter += 1
            batch.append(line)
            if (counter % batch_size) == 0:
                with Pool(12) as p:
                    r = list(p.imap(word_tokenize, batch))
                print("Flattening")
                x = flatten(r)
                print("Counter")
                big_collection += Counter(x)
                print("Resetting")
                batch = []
            if (counter % 1000000) == 0:
                print("Temporary save")
                Serialization.save_obj(big_collection, "stance_word_frequencies")
    Serialization.save_obj(big_collection, "stance_word_frequencies")
    




if __name__ == "__main__":
    # get_word_by_frequency()
    parser = argparse.ArgumentParser(description="Find similar stancemarkers from Wang2vec.")
    parser.add_argument("--embed_path", type=str, help="Path to wav2vec embedding file.", default="wang2vec/embedding_file")
    parser.add_argument("--stance_path", type=str, help="Path to json file of stancemarkers.", default="../stancemarkers/stancemarkers.json")
    parser.add_argument("--cutoff", type=float, help="Cosine similarity cutoff value.", default=0.75)
    parser.add_argument("--out_path", type=str, help="Path to save expanded lexicon.", default="../stancemarkers/expanded_lexicon.json")

    args = parser.parse_args()

    print(args.out_path)
    print("Loading model...")
    w2v_model = load_wang2vec_model(args.embed_path)
    print("Loading stance markers...")
    stance_markers = load_stancemarkers(args.stance_path)

    print("Finding similar stancemarkers...")
    similar_markers = find_similar_stancemarkers(w2v_model, stance_markers, args.out_path, cutoff=args.cutoff)
    # print("Saving markers...")
    # expanded_stancemarkers_to_json(similar_markers, args.out_path)
