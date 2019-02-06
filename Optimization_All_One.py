# coding: utf-8

# # Bayesian Optimization on Keras

# ### MNIST training on Keras with Bayesian optimization
# * This notebook runs MNIST training on Keras using Bayesian optimization to find the best hyper parameters.
# * The MNIST model here is just a simple one with one input layer, one hidden layer and one output layer, without convolution.
# * Hyperparameters of the model include the followings:
# * - number of convolutional layers in first layer
# * - dropout rate of first layer
# * - number of convolutional layers in second layer
# * - dropout rate of second layer
# * - number of units in the Dense Layer
# * - dropout rate of the third layer
# * - batch size
# * - epochs
# * I used GPy and GPyOpt to run Bayesian optimization.


# #### Import libraries

# In[1]:
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, BatchNormalization, Dense
from keras.models import Sequential

import GPy, GPyOpt
import numpy as np
import pandas as pds
import random

from keras.datasets import mnist
from keras.metrics import categorical_crossentropy
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import keras

import os

import pandas as pd


# #### Define MNIST model
# * includes data loading function, training function, fit function and evaluation function

# In[2]:


# MNIST class
class _Optimization():

    def __init__(self, l1_dense_layers=300,
                 batch_size=32, validation_split=0.3):
                 #epochs=5):
        self.l1_dense_layers = l1_dense_layers
        self.l2_dense_layers = l1_dense_layers
        self.l3_dense_layers = l1_dense_layers
        self.l4_dense_layers = l1_dense_layers
        self.l5_dense_layers = l1_dense_layers
        self.batch_size = int(batch_size)
        self.epochs = 100
        self.validation_split = validation_split

        self.results = None


        self.train_x, self.train_y, self.test_x, self.n_cols = self.data()

        self.__model = self._model()

    # load data
    def data(self):

        train_df = pd.read_csv('Football CSV Stats/Master.csv')

        # print(train_df.head())

        train_X = train_df.drop(columns=['Pt_Diff'])

        # print(train_X.head())

        train_y = train_df[['Pt_Diff']]

        test_X = pd.read_csv('Football CSV Stats/TestData.csv')

        n_cols = train_X.shape[1]

        return train_X, train_y, test_X, n_cols

    def _model(self):

        model = Sequential()

        model.add(Dense(self.l1_dense_layers, activation='relu', input_shape=(self.n_cols,)))
        model.add(Dense(self.l2_dense_layers, activation='relu'))
        model.add(Dense(self.l3_dense_layers, activation='relu'))
        model.add(Dense(self.l4_dense_layers, activation='relu'))
        model.add(Dense(self.l5_dense_layers, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    # fit mnist model
    def _fit(self):


        return self.__model.fit(self.train_x, self.train_y, batch_size=self.batch_size,
                                validation_split=self.validation_split, epochs=100, )


        '''
        self.__model.fit(self.__x_train, self.__y_train,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         verbose=0,
                         validation_split=s+elf.validation_split,
                         callbacks=[early_stopping])
                         '''

    # evaluate mnist model
    def _evaluate(self):

        self.results = self._fit()

        evaluation = self.__model.evaluate(self.train_x, self.train_y, batch_size=int(self.batch_size), verbose=1)
        return evaluation


# #### Runner function for the MNIST model

# In[3]:


# function to run mnist class
def run_(validation_split=0.3, l1_dense_layers=300, batch_size=32):


    _optimize = _Optimization(l1_dense_layers=l1_dense_layers, batch_size=int(batch_size),
                                    validation_split=validation_split) #epochs=epochs)


    _evaluation = _optimize._evaluate()

    return _evaluation

# bounds for hyper-parameters in  model
# the bounds dict should be in order of continuous type and then discrete type

bounds = [{'name': 'validation_split', 'type': 'continuous', 'domain': (0.2, 0.8)},
          {'name': 'l1_dense_layers', 'type': 'discrete', 'domain': (500, 1000, 1500, 2000, 2500, 3000, 3500)},
          {'name': 'batch_size', 'type': 'discrete', 'domain': (20, 32, 64, 100, 125,150)},]
         # {'name': 'epochs', 'type': 'discrete', 'domain': (5, 10, 20)}]


# #### Bayesian Optimization

# In[5]:


# function to optimize mnist model
def f(x):
    print(x)
    evaluation = run_(
        validation_split=float(x[:, 0]),
        l1_dense_layers=int(x[:, 1]),
        batch_size=int(x[:, 2]),)
       # epochs=int(x[:, 7]))

    print(" ")
    print("LOSS:\t{0} \t".format(evaluation))
    #print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0], evaluation[1]))
    print(" ")

    print("variables used for this test are:")

    print("validation split: " , float(x[:, 0]))
    print("l1_dense_layers : ", int(x[:, 1]))
    print("batch_size : ", int(x[:, 2]))

    print(" ")


    return evaluation
    #return evaluation[0]


# #### Optimizer instance

# In[6]:


# optimizer
opt_model = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)

# #### Running optimization

# In[7]:


# optimize mnist model
opt_model.run_optimization(max_iter=10)

# #### The output

# In[20]:


# print optimized mnist model
print("""
Optimized Parameters:
\t{0}:\t{1}
\t{2}:\t{3}
\t{4}:\t{5}
""".format(bounds[0]["name"],opt_model.x_opt[0],
           bounds[1]["name"],opt_model.x_opt[1],
           bounds[2]["name"],opt_model.x_opt[2],))


print("optimized loss: {0}".format(opt_model.fx_opt))


# In[21]:


opt_model.x_opt

d = {'Loss': [opt_model.fx_opt], 'Validation Split': [opt_model.x_opt[0]], 'Layer 1': [opt_model.x_opt[1]],
         'Layer 2': [opt_model.x_opt[1]],
         'Layer 3': [opt_model.x_opt[1]], 'Layer 4': [opt_model.x_opt[1]], 'Layer 5': [opt_model.x_opt[1]],
         'Batch Size': [opt_model.x_opt[2]]}



finalVals = pd.DataFrame(data=d)

current = pd.read_csv('Optimization_Runs.csv')

frames = [current, finalVals]

result = pd.concat(frames)

result.to_csv('Optimization_Runs.csv', index=False)