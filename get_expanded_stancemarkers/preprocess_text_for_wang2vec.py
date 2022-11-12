import argparse
import pandas as pd

# Keeping pandas from truncating long strings
pd.set_option('display.max_colwidth', None)

def csvs_to_wang2vec_text(csv_paths, save_path):
    """Given a list to meta data csvs of Reddit data. Extract the body of each post and put
    it on it's own line of a .txt file for training wang2vec."""    
    with open(save_path, "w") as reddit_dataset:
        for csv in csv_paths:
            metadata = pd.read_csv(csv)
            reddit_dataset.write(metadata["body"].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Reddit metadata csv into a text format suitable for Wang2vec.")
    parser.add_argument("--files", nargs='+', default=[], help="A list of metadata .csv files to turn to a .txt")

    args = parser.parse_args()

    save_path = "reddit_dataset.txt"
    csvs_to_wang2vec_text(args.files, save_path)
