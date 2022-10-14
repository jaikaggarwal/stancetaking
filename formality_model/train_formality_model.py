import torch
import pickle
from model import BetaRegression, SBERT, Config
from preprocess import load_formality_dataset_to_df


if __name__ == "__main__":
    torch.cuda.set_device(2)
    data = load_formality_dataset_to_df()
    formality_scores, text = data["avg_formality"], data["text"]

    word_embedder = SBERT("bert-large-nli-mean-tokens")
    embeddings = word_embedder.get_embeddings(text)

    config = Config(epochs=None, k=10, seed=42, stratified=True, bins=6)
    formality_model = BetaRegression(embeddings, formality_scores, config)
    formality_model.train_model()

    filename = "models/formality_model.sav"
    pickle.dump(formality_model.best_model, open(filename, 'wb'))
