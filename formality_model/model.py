import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
import statsmodels.api as sm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score
import tqdm


class Config:
    """Config class used to store hyperparameters for models."""
    def __init__(self, epochs, k, stratified, bins, seed):
        self.epochs = epochs
        self.k = k
        self.stratified = stratified
        self.bins = bins
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
        self.history = {}

    
    def get_score_groups(self, number_of_groups) -> None:
        """Split score into specified bins for stratification."""
        labels = ["a", "b", "c", "d", "e", "f"]
        intervals = pd.cut(self.scores, bins=number_of_groups, labels=labels)

        return intervals
    
    def get_kfolds(self):
        """Return appropriate KFold generator based on config."""
        if self.config.stratified:
            kf = StratifiedKFold(n_splits=self.config.k,
                                 random_state=self.config.seed,
                                 shuffle=True)
        else:
            kf = KFold(n_splits=self.config.k,
                       random_state=self.config.seed,
                       shuffle=True)
        
        return kf

    def train_model(self):

        kf = self.get_kfolds()

        curr_best_fit = -1
        curr_best_model = None

        score_groups = self.get_score_groups(self.config.bins)

        for k, (train_idx, test_idx) in enumerate(kf.split(X=self.scores, y=score_groups)):
            train, test = self.embeddings, self.scores
            X_train = train[train_idx]
            y_train = train.iloc[train_idx, :]

            X_test = test[test_idx]
            y_test = test.iloc[test_idx, :]

            binom_glm = sm.GLM(y_train, X_train, family=sm.families.Binomial())
            binom_fit_model = binom_glm.fit()
            fit_val = self.goodness_of_fit(binom_fit_model, y_test, X_test)

            self.history[k] = fit_val
            self.plot_predictions(binom_fit_model, y_test, X_test, k)

            if fit_val > curr_best_fit:
                print("NEW BEST MODEL")
                print(f"R^2 score of: {fit_val}")
                curr_best_fit = fit_val
                curr_best_model = binom_fit_model
        
        self.best_fit = curr_best_fit
        self.best_model = curr_best_model

        return curr_best_model

    
    def get_goodness_of_fit(self, model, y_true, X):
        """Return R^2 value of model to compare goodness of fit."""
        y_predicted = model.get_prediction(X)
        pred_vals = y_predicted.summary_frame()["mean"]
        fit_val = r2_score(y_true, pred_vals)
        return fit_val

    
    def plot_predictions(self, model, y_true, X, k):
        """Plot model formality predictions vs. actual predictions."""
        y_predicted = model.get_prediction(X)
        pred_vals = y_predicted.summary_frame()["mean"]

        # Unscale values before plotting
        pred_vals = pred_vals * (6) + 3
        y_true = y_true * 6 + 3
        

        plt.scatter(y_true, pred_vals, alpha=0.8)
        plt.xlabel("Actual formality score")
        plt.ylabel("Predicted formality score")
        plt.title(f"{k}-folds Formality model predictions R^2 {self.history[k]}")

        plt.savefig(f"figs/{k}_formality_model_predictions.png")


    # TO DO: Get sentence associated with test set and their predictions
    def get_sentences(self, model, y_true, X, text):
        """Create DataFrame of sentences and their predictions."""
        y_predicted = model.get_prediction(X)
        pred_vals = y_predicted.summary_frame()["mean"]

        pass
