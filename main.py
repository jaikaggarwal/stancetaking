from data_extraction import corpus_statistics as cs
from data_extraction import extract_embeddings_utils as eeu
from data_extraction import feature_extraction as fe
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool


def main(dir_name, num_cores):
    # Step 0: Load files and prepare directories
    #TODO: Use exisiting pipeline data_dir to avoid repetition of this part of the pipeline
    ROOT_DIR = f"/ais/hal9000/datasets/reddit/stance_pipeline/{dir_name}/"
    RAW_DATA_DIR = "/ais/hal9000/datasets/reddit/stance_analysis/"
    TEST_COMMUNITIES_FILE = "data_extraction/test_communities.csv"
    PIPELINE_DATA_DIR = ROOT_DIR + "current_data/"
    EMBEDDINGS_DIR = ROOT_DIR + "embeddings/"
    SITUATIONS_DIR = ROOT_DIR + "full_features/"

    files = sorted(list(os.walk(RAW_DATA_DIR)))
    data_dirs = sorted([dir_tup[0] for dir_tup in files if dir_tup[0].endswith("files")])
    
    for dir_str in [ROOT_DIR, PIPELINE_DATA_DIR, EMBEDDINGS_DIR, SITUATIONS_DIR]:
        if not os.path.exists(dir_str):
            print(f"Creating directory: {dir_str}")
            os.makedirs(dir_str)
        else:
            print(f"Using existing directory: {dir_str}")
    
    # Step 1: Gather all data from test communities #TODO: Write as wrapper
    print("Extracting community data...")
    with Pool(num_cores) as p:
        r = list(tqdm.tqdm(p.imap(lambda x: cs.extract_test_data(TEST_COMMUNITIES_FILE, x, PIPELINE_DATA_DIR), data_dirs), total=len(data_dirs)))
    
    # Step 2: Create SBERT model
    print("Loading SBERT model...")
    sbert_model = eeu.SBERT("bert-large-nli-mean-tokens")
    
    # Step 3: Create embeddings and metadata file (cannot parallelize, based on GPUs)
    print("Create embeddings...")
    eeu.embeddings_wrapper(PIPELINE_DATA_DIR, EMBEDDINGS_DIR, sbert_model)
    
    # Step 4: Infer emotional values to create semantic situations
    print("Extracting features...")
    fe.extraction_wrapper(EMBEDDINGS_DIR, SITUATIONS_DIR, num_cores)

    #TODO:
    # Step 5: Bin/cluster semantic situations
    # Step 6: Compute SVD


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stancetaking pipeline")
    parser.add_argument("--num_cores", type=int, help="Number of cores to use.", default=6)
    parser.add_argument("output_dir", type=str, help="Where to store data")
    
    args = parser.parse_args()