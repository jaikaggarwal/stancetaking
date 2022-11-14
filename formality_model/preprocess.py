import pandas as pd

def load_formality_data(path_to_data):
    """Load data from formality dataset.

    Args:
        path_to_data (String): Path to file.

    Returns:
        DataFrame: Pandas DataFrame containing formality data.
    """

    # Read data
    df = pd.read_csv(path_to_data, sep="\t", encoding="utf-8")

    # Keep relevant columns
    df = df.iloc[:, [0, 3]]
    df.columns = ["avg_formality", "text"]
    
    # Minmax rescale between 0 and 1
    df["avg_formality"] = rescale_formality(df["avg_formality"])

    return df


def rescale_formality(col):
    """Rescale a column between 0 and 1 for BetaRegression."""
    rescaled = (col - col.min()) / (col.max() - col.min())
    return rescaled

def formality_dataset_generator(datasets=None):
    # Read each type of data
    if datasets is None:
        datasets = ["answers", "blog", "email", "news"]
    
    for dataset in datasets:
        yield load_formality_data(f"data/data-for-release/{dataset}")

def load_formality_dataset_to_df(datasets=None):
    # Stitch data into DataFrame
    df = pd.concat(list(formality_dataset_generator(datasets)), ignore_index=True)
    return df

if __name__ == "__main__":
    datasets = ["answers", "blog"]
    df = load_formality_dataset_to_df(datasets)
