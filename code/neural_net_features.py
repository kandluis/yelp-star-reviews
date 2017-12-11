# coding: utf-8
# Use features from the network to train some cool stuff.
import os

import numpy as np
import pandas as pd
import snap
import pickle

from sklearn.manifold import TSNE

# Read datasets.
DATA_DIR = "../yelp_data/dataset"
OUTPUT_DIR = "../shared/figures"

with open(os.path.join(DATA_DIR, "test_features.pkl")) as ftest,      open(os.path.join(DATA_DIR, "test_rating.pkl")) as rtest:
  (_, testFeats) = pickle.load(ftest)
  testY = pickle.load(rtest)

visualization = TSNE(n_components=2, perplexity=50)

embedding = visualization.fit_transform(testFeats)
