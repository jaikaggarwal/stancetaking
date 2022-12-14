import argparse
from nltk.tokenize import sent_tokenize
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from sentence_transformers.SentenceTransformer import torch as pt
pt.cuda.set_device(1)

INPUT_DIR = "/ais/hal9000/datasets/reddit/stance_analysis/test_run_data/"
OUTPUT_DIR = "/ais/hal9000/datasets/reddit/jai_stance_embeddings/"

class SBERT:
    """Wrapper class for SBERT, used to encode text."""
    def __init__(self, model_name) -> None:
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, data):
        embeddings = self.model.encode(data, show_progress_bar=True)
        return embeddings


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
    all_text_embeddings = sbert.get_embeddings(data.body)

    # Create dataframe for post-level embeddings
    all_text_df = pd.DataFrame(all_text_embeddings)
    all_text_df = pd.concat([data.id, all_text_df], axis=1)

    all_text_df.to_csv(post_level_embeds_output_dir, index=False)

    # post_lengths = np.array([])
    # post_ids = np.array([])
    # sentence_ids = np.array([])
    # sentence_embeds = []
    
    # # Create dataframe for sentence-level embeddings
    # # Iterate by post
    # for post in data.itertuples():
    #     # Get sentences using nltk sent_tokenize
    #     text = post.body
    #     post_sentences = sent_tokenize(text)
    #     # Save post length for metadata
    #     post_lengths = np.append(post_lengths, len(post_sentences))

    #     for i, sent in enumerate(post_sentences):
    #         sentence_embed = sbert.get_embeddings(sent)
    #         sentence_embeds.append(sentence_embed)
    #         post_ids = np.append(post_ids, post.id)
    #         sentence_ids = np.append(sentence_ids, i)
    
    # sentence_embeds = np.vstack(sentence_embeds)
    # sentence_df = pd.DataFrame(sentence_embeds)
    # post_ids = pd.Series(post_ids, name="id")
    # sentence_ids = pd.Series(sentence_ids, name="sentence_id")
    
    # all_sent_df = pd.concat([post_ids, sentence_ids, sentence_df], axis=1)
    # all_sent_df.to_csv(sentence_level_embeds_output_dir, index=False)
            

    # Create metadata csv
    # post_lengths = pd.Series(post_lengths, name="sentence_count")
    metadata = data[["id", "author", "subreddit", "body", "created_utc", "rel_marker", "marker_category", "len"]]
    # metadata = pd.concat([metadata, post_lengths], axis=1)
    metadata.to_csv(metadata_output_dir, index=False)

if __name__ == "__main__":
    sbert_model = SBERT("bert-large-nli-mean-tokens")
    files = os.listdir(INPUT_DIR)
    for file in tqdm(files):
        print(file)
        process_datadumps(INPUT_DIR + file, sbert_model, OUTPUT_DIR + file[:-4] + "_embeddings.csv", OUTPUT_DIR  + file[:-4] +  "_metadata.csv")