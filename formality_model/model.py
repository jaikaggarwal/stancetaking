import pandas as pd
from sentence_transformers import SentenceTransformer
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import tqdm


class Config:
    """Config class used to store hyperparameters for models."""
    def __init__(self, epochs, k, seed):
        self.epochs = epochs
        self.k = k
        self.seed = seed

class SBERT:
    """Wrapper class for SBERT, used to encode text."""
    def __init__(self, model_name) -> None:
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, data):
        embeddings = self.model.encode(data, show_progress_bar=True)
        return embeddings


class BetaRegression:
    def __init__(self, embeddings, scores, config) -> None:
        self.embeddings = embeddings
        self.scores = scores
        self.config = config
        self.best_model = None
        self.best_fit = -1
    
    def train_model(self):
        kf = KFold(n_splits=self.config.k,
                   random_state=self.config.seed,
                   shuffle=True)

        curr_best_fit = -1
        curr_best_model = None

        for train_idx, test_idx in kf.split(X=self.scores):
            train, test = self.embeddings, self.scores
            X_train = train[train_idx]
            y_train = test.iloc[train_idx, :]

            X_test = test[test_idx]
            y_test = test.iloc[test_idx, :]

            binom_glm = sm.GLM(y_train, X_train, family=sm.families.Binomial())
            binom_fit_model = binom_glm.fit()
            fit_val = self.goodness_of_fit(binom_fit_model, y_test, X_test)

            if fit_val > curr_best_fit:
                print("NEW BEST MODEL")
                print(f"R^2 score of: {fit_val}")
                curr_best_fit = fit_val
                curr_best_model = binom_fit_model
        
        self.best_fit = curr_best_fit
        self.best_model = curr_best_model

        return curr_best_model

    
    def get_goodness_of_fit(model, y_true, X):
        """Return R^2 value of model to compare goodness of fit."""
        y_predicted = model.get_prediction(X)
        pred_vals = y_predicted.summary_frame()['mean']
        fit_val = r2_score(y_true, pred_vals)
        return fit_val
