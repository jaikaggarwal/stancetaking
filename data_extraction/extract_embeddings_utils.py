import argparse
from nltk.tokenize import sent_tokenize
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

from sentence_transformers.SentenceTransformer import torch as pt
pt.cuda.set_device(1)

INPUT_DIR = "/ais/hal9000/datasets/reddit/stance_analysis/test_run_data/"
OUTPUT_DIR = "/ais/hal9000/datasets/reddit/jai_stance_embeddings/"

class SBERT:
    """Wrapper class for SBERT, used to encode text."""
    def __init__(self, model_name="bert-large-nli-mean-tokens") -> None:
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, data):
        pt.cuda.empty_cache()
        embeddings = self.model.encode(data, show_progress_bar=True)
        return embeddings

    def get_tokenize_lengths(self, data, remove_special_tokens=True):
        """Get number of token segments in the SBERT tokenization of sentences.
        
        Optionally include or exclude the special <CLS> and <SEP> tokens that are
        added automatically.
        """
        # If this is just a raw string, nest it in a list
        if type(data) == str:
            data = [data]
        tokenizations = self.model.tokenize(data)["input_ids"]

        # Since padding tokens are tokenized with id 0, we just count
        # non zero elements.

        # If we have more than 1 post, vectorize counting over each post.
        if tokenizations.dim() > 1:
            counts = tokenizations.count_nonzero(axis=1)
        else:
            counts = tokenizations.count_nonzero()

        if remove_special_tokens:
            # Remove the 2 special tokens
            counts = counts - 2
        return counts


def process_datadumps(dump_file, sbert, post_level_embeds_output_dir, metadata_output_dir):
    """Process Reddit data dumps csv into metadata and SBert embeddings.
    
    The embeddings are separated into post-level embeddings where each post
    gets a singular embedding as calculated by one encoding of SBERT and
    sentence-level embeddings where each sentence in a post gets its own
    separate embedding."""

    data = pd.read_csv(dump_file)
    data['sen_id'] = data.groupby("id").cumcount()
    data['id'] = data['id'] + "-" +  data['sen_id'].astype(str)

    # Create post-level embeddings
    NUM_SPLITS = 40
    data_splits = np.array_split(data, NUM_SPLITS)
    all_text_dfs = []
    for i, split in enumerate(data_splits):
        split = split.reset_index(drop=True)
        all_text_embeddings = sbert.get_embeddings(split.body_mask.tolist())
        all_text_df = pd.DataFrame(all_text_embeddings)
        all_text_df = pd.concat([split.id, all_text_df], axis=1)
        # all_text_dfs.append(all_text_df)
        all_text_df.to_csv(post_level_embeds_output_dir + f"_00{i}.csv", index=False)            

    # Create metadata csv
    # post_lengths = pd.Series(post_lengths, name="sentence_count")
    metadata = data[["id", "author", "subreddit", "body", "created_utc", "rel_marker", "len"]]
    # metadata = pd.concat([metadata, post_lengths], axis=1)
    metadata.to_csv(metadata_output_dir, index=False)

def embeddings_wrapper(input_dir, output_dir, sbert_model, range_start, range_finish):
    files = sorted(os.listdir(input_dir))[range_start:range_finish]
    print(files)
    for file in tqdm(files):
        print(file)
        process_datadumps(
            input_dir + file, 
            sbert_model, 
            output_dir + file[:-4] + "_embeddings", 
            output_dir + file[:-4] + "_metadata.csv"
        )

if __name__ == "__main__":
    sbert_model = SBERT("bert-large-nli-mean-tokens")
    files = os.listdir(INPUT_DIR)
    for file in tqdm(files):
        print(file)
        process_datadumps(INPUT_DIR + file, sbert_model, OUTPUT_DIR + file[:-4] + "_embeddings.csv", OUTPUT_DIR  + file[:-4] +  "_metadata.csv")