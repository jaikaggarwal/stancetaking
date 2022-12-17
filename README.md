# Stancetaking Repository
Our data extraction pipeline runs from the main.py file. This script relies on three files within the data_extraction folder.

1. Corpus_statistics.py: this file is used to extract our posts from the Reddit data dumps, and also filters the data according to the preprocessing steps mentioned in our paper.
2. Extract_embedding_utils.py: this file contains utility functions that create SBERT embeddings for each sentence in our dataset
3. Feature_extraction.py: this file is used to extract our linguistic features using saved regression models

We then use semantic_spaces/test_clustering.ipynb to perform the remaining analyses, including extracting the latent dimensions, performing our validation, and the analyses for research questions 1 and 2. The code for each can be found under corresponding headers in the notebook.
