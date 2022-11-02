import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import sys
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from utils import Serialization
from sentence_transformers.SentenceTransformer import torch as pt
pt.cuda.set_device(1)
print(pt.cuda.is_available())
print(pt.__version__)
tqdm.pandas()
model = SentenceTransformer("bert-large-nli-mean-tokens")

short_posts = pd.read_csv("short_posts.csv", index_col=0)
mid_posts = pd.read_csv("mid_posts.csv", index_col=0)
long_posts = pd.read_csv("long_posts.csv", index_col=0)

valence_model = Serialization.load_obj('valence_model')
arousal_model = Serialization.load_obj('arousal_model')
dominance_model = Serialization.load_obj('dominance_model')

def infer_emotion_value(post, regressor_v, regressor_a, regressor_d):
    try:
        embeddings = model.encode(post, show_progress_bar=False)
        v_predictions = regressor_v.predict(embeddings)
        a_predictions = regressor_a.predict(embeddings)
        d_predictions = regressor_d.predict(embeddings)
        # assert(len(v_predictions) == len(embeddings))
        assert type(np.mean(v_predictions)) == np.float64
        return v_predictions[0], a_predictions[0], d_predictions[0]
        # return [np.mean(v_predictions), np.mean(a_predictions), np.mean(d_predictions)]
    except Exception as e:
        print(e)
        return [np.nan, np.nan, np.nan]


for i in [6, 8, 10, 12, 14, 16]:
    curr = pd.read_csv(f"valence_analysis/{i}_len_posts.csv")
    curr["output"] = curr['body'].progress_apply(lambda x: infer_emotion_value(x, valence_model, arousal_model, dominance_model))
    curr["output_masked"] = curr['body_mask'].progress_apply(lambda x: infer_emotion_value(x, valence_model, arousal_model, dominance_model))
    curr.to_csv(f"valence_analysis/{i}_len_posts_vad.csv")