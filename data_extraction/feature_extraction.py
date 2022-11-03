import re
import sys, csv
from random import shuffle
import numpy as np
from sentence_transformers import SentenceTransformer
import statsmodels.api as sm
from sklearn.metrics import r2_score
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import os

from utils import Serialization, pickle

from sentence_transformers.SentenceTransformer import torch as pt
pt.cuda.set_device(1)
print(pt.cuda.is_available())
print(pt.__version__)
model = SentenceTransformer('bert-large-nli-mean-tokens')

tqdm.pandas() 

HELD_OUT_PROP = 0.8
DATA_DIR = '/u/jai/AutismMarkers/data_files/'


class LexicalAnalysis():

    @staticmethod
    def infer_emotion_value(embeddings, regressor_v, regressor_a, regressor_d):
        pt.cuda.empty_cache()
        try:
            v_predictions = regressor_v.predict(embeddings)
            a_predictions = regressor_a.predict(embeddings)
            d_predictions = regressor_d.predict(embeddings)
            assert(len(v_predictions) == len(embeddings))
            assert type(np.mean(v_predictions)) == np.float64
            return [v_predictions, a_predictions, d_predictions]
        except Exception as e:
            print(e)
            return [np.nan, np.nan, np.nan, np.nan]


    #### SBERT-related Methods #### 
    @staticmethod
    def fit_beta_reg(y, X, df, group_title):
       
        curr_best_fit = 0
        curr_best_model = None

        for i in tqdm(range(10)):

            X_sample = df.groupby(group_title).apply(lambda temp: temp.sample(int(HELD_OUT_PROP*len(temp))))
            train_idx = pd.Series(X_sample.index.get_level_values(1))
            test_idx = df.index.difference(train_idx).tolist()
            np.random.shuffle(test_idx)
            train_idx = train_idx.tolist()

            # print(train_idx)
            # print(len(train_idx))
            # print(len(test_idx))
            # print(len(set(train_idx).union(set(test_idx))))
            # print(X.shape)
            # print(df.shape)

            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            binom_glm = sm.GLM(y_train, X_train, family=sm.families.Binomial())
            binom_fit_model = binom_glm.fit()
            fit_val = LexicalAnalysis.goodness_of_fit(binom_fit_model, y_test, X_test)

            if fit_val > curr_best_fit:
                print("NEW BEST MODEL")
                print(f"R^2 score of: {fit_val}")
                curr_best_fit = fit_val
                curr_best_model = binom_fit_model
        return curr_best_model
    
    @staticmethod
    def goodness_of_fit(model, true, X):
        y_predicted = model.get_prediction(X)
        pred_vals = y_predicted.summary_frame()['mean']
        fit_val = r2_score(true, pred_vals)
        print(fit_val)
        return fit_val

    
    #### Lexical Analysis Functions ####
    @staticmethod
    def get_embeddings(data, title):
        try:
            embeddings = Serialization.load_obj(title)
        except FileNotFoundError:
            embeddings = model.encode(data, show_progress_bar=True)
            Serialization.save_obj(embeddings, title)
        return embeddings

    def init_politeness():
        politeness_clf = Serialization.load_obj('wikipedia_politeness_classifier')
        return politeness_clf

    def init_formality():
        model = pickle.load(open("../formality_model/models/formality_model.sav", "rb"))
        return model
    
    def infer_formality(embeddings, formality_model):
        probs = formality_model.predict(embeddings)
        return probs

    def infer_politeness(embeddings, politeness_clf):
        probs = politeness_clf.predict_proba(embeddings)
        offset = probs + 0.0000001
        log_odd_vals = np.log(np.maximum(offset/(1-offset), 0.0000001))
        return log_odd_vals[:, 1]

    @staticmethod
    def init_vad():
        df_vad = pd.read_csv('/ais/hal9000/jai/lexicon.txt', delimiter='\t', header=0)
        df_vad = df_vad.dropna().reset_index(drop=True)
        df = df_vad[['Word', 'Valence', 'Arousal', 'Dominance']]
        valence = np.array(df['Valence'].tolist())
        arousal = np.array(df['Arousal'].tolist())
        dominance = np.array(df['Dominance'].tolist())

        df['valence_scaled'] = df['Valence']*4 + 1
        df['arousal_scaled'] = df['Arousal']*4 + 1
        df['dominance_scaled'] = df['Dominance']*4 + 1

        df['v_group'] = df['valence_scaled'].apply(np.ceil)
        df['a_group'] = df['arousal_scaled'].apply(np.ceil)
        df['d_group'] = df['dominance_scaled'].apply(np.ceil)

        
        vad_words = list(df_vad['Word'])

        vad_embeddings = LexicalAnalysis.get_embeddings(vad_words, "vad")

        print("LOADING VALENCE MODEL")
        try:
            valence_model = Serialization.load_obj('valence_model')
        except FileNotFoundError:
            valence_model = LexicalAnalysis.fit_beta_reg(valence, vad_embeddings, df, 'v_group')
            Serialization.save_obj(valence_model, 'valence_model')

        print("LOADING AROUSAL MODEL")
        try:
            arousal_model = Serialization.load_obj('arousal_model')
        except FileNotFoundError:
            arousal_model = LexicalAnalysis.fit_beta_reg(arousal, vad_embeddings, df, 'a_group')
            Serialization.save_obj(arousal_model, 'arousal_model')

        print("LOADING DOMINANCE MODEL")
        try:
            dominance_model = Serialization.load_obj('dominance_model')
        except FileNotFoundError:
            dominance_model = LexicalAnalysis.fit_beta_reg(dominance, vad_embeddings, df, 'd_group')
            Serialization.save_obj(dominance_model, 'dominance_model')
        
        LexicalAnalysis.goodness_of_fit(valence_model, valence, vad_embeddings)
        LexicalAnalysis.goodness_of_fit(arousal_model, arousal, vad_embeddings)
        LexicalAnalysis.goodness_of_fit(dominance_model, dominance, vad_embeddings)

        return valence_model, arousal_model, dominance_model
    

if __name__ == "__main__":

    INPUT_DIR = "/ais/hal9000/datasets/reddit/jai_stance_embeddings/"

    data = os.listdir(INPUT_DIR)
    embedding_files = sorted([file for file in data if "embeddings.csv" in file])
    metadata_files = sorted([file for file in data if "metadata.csv" in file])

    valence_model, arousal_model, dominance_model = LexicalAnalysis.init_vad()
    politeness_model = LexicalAnalysis.init_politeness()
    formality_model = LexicalAnalysis.init_formality()

    for e, m in tqdm(zip(embedding_files, metadata_files), total=len(embedding_files)):
        e_file = pd.read_csv(INPUT_DIR + e, index_col='id')
        m_file = pd.read_csv(INPUT_DIR + m, index_col='id')
        embeddings = e_file.to_numpy()
        vad_vals = LexicalAnalysis.infer_emotion_value(embeddings, valence_model, arousal_model, dominance_model)
        vad_vals = pd.DataFrame(np.asarray(vad_vals).T, index=m_file.index, columns=['Valence', 'Arousal', 'Dominance'])
        m_file = pd.concat((m_file, vad_vals), axis=1)
        m_file['Politeness'] = LexicalAnalysis.infer_politeness(embeddings, politeness_model)
        m_file['Formality'] = LexicalAnalysis.infer_formality(embeddings, formality_model)
        m_file.to_csv(INPUT_DIR + m[:-4] + "_vad.csv")