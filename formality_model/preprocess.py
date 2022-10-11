import pandas as pd

def load_formality_data(path_to_data):
    """Load data from formality dataset.

    Args:
        path_to_data (String): Path to file.

    Returns:
        DataFrame: Pandas DataFrame containing formality data.
    """
    df = pd.read_csv(path_to_data, sep="\t", encoding="utf-8")
    df = df.iloc[:, [0, 3]]
    df.columns = ["avg_formality", "text"]
    return df

def formality_dataset_generator():
    datasets = ["answers", "blog", "email", "news"]
    for dataset in datasets:
        yield load_formality_data(f"data/data-for-release/{dataset}")

def load_formality_dataset_to_df():
    df = pd.concat(list(formality_dataset_generator()), ignore_index=True)
    return df

if __name__ == "__main__":
    df = load_formality_dataset_to_df()
    