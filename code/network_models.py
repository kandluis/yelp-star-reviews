# coding: utf-8
import os

import numpy as np
import pandas as pd
import snap
import pickle

from sklearn.ensemble import RandomForestClassifier

# Read datasets.
DATA_DIR = "../yelp_data/dataset"
OUTPUT_DIR = "../shared/figures"

with open(os.path.join(DATA_DIR, "val_features.pkl")) as fval,      open(os.path.join(DATA_DIR, "train_features.pkl")) as ftrain,      open(os.path.join(DATA_DIR, "train_rating.pkl")) as rtrain,      open(os.path.join(DATA_DIR, "val_rating.pkl")) as rval:
  (_, valFeats) = pickle.load(fval)
  valY = pickle.load(rval)
  (_, trainFeats) = pickle.load(ftrain)
  trainY = pickle.load(rtrain)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


class BaselineModel(object):

  def fit(self, X, y):
    self._mean = np.mean(y)
    return self

  def predict(self, X):
    return np.array([self._mean for _ in xrange(len(X))])

  def score(self, X, true):
    predicted = self.predict(X)
    u = np.sum((true - predicted)**2)
    v = np.sum((true - np.mean(true))**2)
    return 1 - u/v

  def get_params(self):
    return ""

linearRegressor = LinearRegression()

rideRegressor = Ridge(alpha=1)

bayesianRegressor = BayesianRidge(n_iter=100, compute_score=True)

baselinePredictor = BaselineModel()

neuralNetworkPredictor = MLPRegressor(
    hidden_layer_sizes=(200, 40, 8, 2), max_iter=1000, early_stopping=True)

randomForestPredictor = RandomForestRegressor(n_estimators=1000)


def rmse(predicted, true):
  return np.sqrt(np.sum((predicted - true)**2) / len(true))


def relative_error(predicted, true):
  m = max(np.max(predicted), np.max(true))
  return np.mean(np.abs(predicted - true)) / m


def fitModel(name, model):
  model = model.fit(trainFeats, trainY)
  predicted = model.predict(valFeats)
  print "Score for %s is %s" % (
      name, model.score(valFeats, valY))
  print "RMSE for %s is %s" % (
      name, rmse(predicted, valY))
  print "Average relative error for %s is %s percent." % (
      name, 100*relative_error(predicted, valY))

fitModel("baseline", baselinePredictor)

fitModel("linear regression", linearRegressor)

fitModel("ridge regression", rideRegressor)

fitModel("bayesian regression", bayesianRegressor)

fitModel("neural network", neuralNetworkPredictor)

fitModel("random forest", randomForestPredictor)

with open(os.path.join(DATA_DIR, "test_features.pkl")) as ftest,      open(os.path.join(DATA_DIR, "test_rating.pkl")) as rtest:
  (_, testFeats) = pickle.load(ftest)
  testY = pickle.load(rtest)

# Now that we have the trained models, test them with the test data.


def finalTestModel(name, model):
  predicted = model.predict(testFeats)
  print "Score for %s is %s" % (
      name, model.score(testFeats, testY))
  print "RMSE for %s is %s" % (
      name, rmse(predicted, testY))
  print "Average relative error for %s is %s percent." % (
      name, 100*relative_error(predicted, testY))

finalTestModel("baseline", baselinePredictor)

finalTestModel("linear regression", linearRegressor)

finalTestModel("ridge regression", rideRegressor)

finalTestModel("bayesian regression", bayesianRegressor)

finalTestModel("neural network", neuralNetworkPredictor)

finalTestModel("random forest", randomForestPredictor)


def testTrainModel(name, model):
  predicted = model.predict(trainFeats)
  print "Score for %s is %s" % (
      name, model.score(trainFeats, trainY))
  print "RMSE for %s is %s" % (
      name, rmse(predicted, trainY))
  print "Average relative error for %s is %s percent." % (
      name, 100*relative_error(predicted, trainY))

testTrainModel("baseline", baselinePredictor)

testTrainModel("linear regression", linearRegressor)

testTrainModel("ridge regression", rideRegressor)

testTrainModel("bayesian regression", bayesianRegressor)

testTrainModel("neural network", neuralNetworkPredictor)

testTrainModel("random forest", randomForestPredictor)
