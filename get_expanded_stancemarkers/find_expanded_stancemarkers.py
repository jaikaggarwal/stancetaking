import argparse
from collections import defaultdict
import json

import numpy as np
from gensim.models import KeyedVectors
from gensim import models

def load_wang2vec_model(model_path):
    return models.KeyedVectors.load_word2vec_format(model_path, binary=True)

def load_stancemarkers(stance_path):
    with open(stance_path) as stancemarkers:
        markers = json.load(stancemarkers)
        return markers


def find_similar_stancemarkers(w2v_model, stancemarkers, cutoff=0.75):
    """Return a dictionary of words that have cosine similarities above some cutoff
    with respect to any of a list of stancemarkers.
    
    The keys are the word and the value is a list of all stancemarkers they are similar
    to.
    """
    expanded_lexicon = defaultdict(set)
    keys = np.asarray(w2v_model.index_to_key)
    for stancemarker in stancemarkers:
        if stancemarker in keys:
            # Find all the word embeddings from wang2vec with cosine similarity
            # greater than cutoff with this stancemarker
            similar_markers = w2v_model.cosine_similarities(w2v_model[stancemarker], w2v_model.vectors) >= cutoff
            
            for similar_marker in keys[similar_markers]:
                expanded_lexicon[similar_marker].add(stancemarkers[stancemarker]["stance_group"])
            
    return expanded_lexicon


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find similar stancemarkers from Wang2vec.")
    parser.add_argument("--embed_path", type=str, help="Path to wav2vec embedding file.", default="wang2vec/embedding_file")
    parser.add_argument("--stance_path", type=str, help="Path to json file of stancemarkers.", default="../stancemarkers/stancemarkers.json")
    parser.add_argument("--cutoff", type=float, help="Cosine similarity cutoff value.", default=0.75)
    parser.add_argument("--out_path", type=str, help="Path to save expanded lexicon.", default="../stancemarkers/expanded_lexicon.json")

    args = parser.parse_args()

    w2v_model = load_wang2vec_model(args.embed_path)
    stance_markers = load_stancemarkers(args.stance_path)

    similar_markers = find_similar_stancemarkers(w2v_model, stance_markers, cutoff=args.cutoff)
    expanded_stancemarkers_to_json(similar_markers, args.out_path)
