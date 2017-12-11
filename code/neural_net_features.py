
# coding: utf-8
# Use features from the network to train some cool stuff.
import os

import numpy as np
import pandas as pd
import snap
import pickle

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

# Read datasets.
DATA_DIR = "../yelp_data/dataset"
OUTPUT_DIR = "../shared/figures"

with open(os.path.join(DATA_DIR, "test_features.pkl")) as ftest,\
        open(os.path.join(DATA_DIR, "test_rating.pkl")) as rtest:
  (_, testFeats) = pickle.load(ftest)
  testY = pickle.load(rtest)

visualization = TSNE(n_components=2, perplexity=50)

embedding = visualization.fit_transform(testFeats)

testY

X = zip(embedding[:, 0], testY)

Y = zip(embedding[:, 1], testY)

X1 = [x for x, r in X if r == 1]
X2 = [x for x, r in X if r == 2]
X3 = [x for x, r in X if r == 3]
X4 = [x for x, r in X if r == 4]
X5 = [x for x, r in X if r == 5]

Y1 = [y for y, r in Y if r == 1]
Y2 = [y for y, r in Y if r == 2]
Y3 = [y for y, r in Y if r == 3]
Y4 = [y for y, r in Y if r == 4]
Y5 = [y for y, r in Y if r == 5]

for X, Y in [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4), (X5, Y5)]:
  plt.scatter(X, Y, alpha=0.1)

plt.title("TSE on Test Dataset")
plt.savefig(os.path.join(OUTPUT_DIR, "TSE.png"), dpi=400)
plt.show()
