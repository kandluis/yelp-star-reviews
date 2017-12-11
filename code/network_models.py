
# coding: utf-8

# In[2]:


import os

import numpy as np
import pandas as pd
import snap
import pickle

from sklearn.ensemble import RandomForestClassifier


# In[3]:


# Read datasets.
DATA_DIR = "../yelp_data/dataset"
OUTPUT_DIR = "../shared/figures"


# In[10]:


with open(os.path.join(DATA_DIR, "val_features.pkl")) as fval,      open(os.path.join(DATA_DIR, "train_features.pkl")) as ftrain,      open(os.path.join(DATA_DIR, "train_rating.pkl")) as rtrain,      open(os.path.join(DATA_DIR, "val_rating.pkl")) as rval:
    (_, valFeats) = pickle.load(fval)
    valY = pickle.load(rval)
    (_, trainFeats) = pickle.load(ftrain)
    trainY = pickle.load(rtrain)


# In[143]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


# In[79]:


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


# In[80]:


linearRegressor = LinearRegression()


# In[90]:


rideRegressor = Ridge(alpha=1)


# In[159]:


bayesianRegressor = BayesianRidge(n_iter=100, compute_score=True)


# In[147]:


baselinePredictor = BaselineModel()


# In[148]:


neuralNetworkPredictor = MLPRegressor(
    hidden_layer_sizes=(200,40,8,2), max_iter=1000, early_stopping=True)


# In[149]:


randomForestPredictor = RandomForestRegressor(n_estimators=1000)


# In[150]:


def rmse(predicted, true):
    return np.sqrt(np.sum((predicted - true)**2) / len(true))


# In[151]:


def relative_error(predicted, true):
    m = max(np.max(predicted), np.max(true))
    return np.mean(np.abs(predicted -true)) / m


# In[152]:


def fitModel(name, model):
    model = model.fit(trainFeats, trainY)
    predicted = model.predict(valFeats)
    print "Score for %s is %s" % (
        name, model.score(valFeats, valY))
    print "RMSE for %s is %s" % (
        name, rmse(predicted, valY))
    print "Average relative error for %s is %s percent." % (
        name, 100*relative_error(predicted, valY))


# In[153]:


fitModel("baseline", baselinePredictor)


# In[154]:


fitModel("linear regression", linearRegressor)


# In[155]:


fitModel("ridge regression", rideRegressor)


# In[166]:


fitModel("bayesian regression", bayesianRegressor)


# In[157]:


fitModel("neural network", neuralNetworkPredictor)


# In[169]:


fitModel("random forest", randomForestPredictor)


# In[160]:


with open(os.path.join(DATA_DIR, "test_features.pkl")) as ftest,      open(os.path.join(DATA_DIR, "test_rating.pkl")) as rtest:
    (_, testFeats) = pickle.load(ftest)
    testY = pickle.load(rtest)


# In[161]:


# Now that we have the trained models, test them with the test data.
def finalTestModel(name, model):
    predicted = model.predict(testFeats)
    print "Score for %s is %s" % (
        name, model.score(testFeats, testY))
    print "RMSE for %s is %s" % (
        name, rmse(predicted, testY))
    print "Average relative error for %s is %s percent." % (
        name, 100*relative_error(predicted, testY))


# In[162]:


finalTestModel("baseline", baselinePredictor)


# In[163]:


finalTestModel("linear regression", linearRegressor)


# In[164]:


finalTestModel("ridge regression", rideRegressor)


# In[167]:


finalTestModel("bayesian regression", bayesianRegressor)


# In[168]:


finalTestModel("neural network", neuralNetworkPredictor)


# In[170]:


finalTestModel("random forest", randomForestPredictor)


# In[171]:


def testTrainModel(name, model):
    predicted = model.predict(trainFeats)
    print "Score for %s is %s" % (
        name, model.score(trainFeats, trainY))
    print "RMSE for %s is %s" % (
        name, rmse(predicted, trainY))
    print "Average relative error for %s is %s percent." % (
        name, 100*relative_error(predicted, trainY))


# In[172]:


testTrainModel("baseline", baselinePredictor)


# In[174]:


testTrainModel("linear regression", linearRegressor)


# In[175]:


testTrainModel("ridge regression", rideRegressor)


# In[177]:


testTrainModel("bayesian regression", bayesianRegressor)


# In[178]:


testTrainModel("neural network", neuralNetworkPredictor)


# In[180]:


testTrainModel("random forest", randomForestPredictor)

