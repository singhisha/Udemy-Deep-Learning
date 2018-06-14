# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, 
                     engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, 
                     engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, 
                     engine = 'python', encoding = 'latin-1')

# Training set and test set
training = pd.read_csv('ml-100k/u1.base', sep = '\t')
training = np.array(training, dtype = 'int')
test = pd.read_csv('ml-100k/u1.test', sep = '\t')
test = np.array(test, dtype = 'int')

# Data convert to format with users in line and movies in column
# We need all the users and all the movies in both tratining and test set
max_users = int(max(max(training[:, 0]), max(test[:,0])))
max_movies = int(max(max(training[:, 1]), max(test[:,1])))
def convert(data):
    new_data = []
    for user in range(1, max_users+1):
        movies = data[:, 1][data[:, 0] == user]
        ratings = data[:, 2][data[:, 0] == user]
        final_ratings = np.zeros(max_movies)
        final_ratings[movies-1] = ratings
        new_data.append(list(final_ratings))
    return new_data

training = convert(training)
test = convert(test)
        
# Convert into Torch tensors
# Torch tensors are muilti-dimensional matrices with a single type
# Can use numpy array but it will be less efficient
training = torch.FloatTensor(training)
test = torch.FloatTensor(test)   

# Converting rating to binary
training[training == 0] = -1
training[training == 1] = 0
training[training == 2] = 0
training[training > 2] = 1
test[test == 0] = -1
test[test == 1] = 0
test[test == 2] = 0
test[test > 2] = 1

# Creating the architecture of the RBM
# Weights are all the parameters of the probability of visible nodes
# given the hidden nodes
# bias for probabilty of hidden nodes given the visible nodes
# bias for probability of visible nodes given the hidden nodes
class RBM:
    def __init__(self, visible, hidden):
        self.weights = torch.randn(visible, hidden)
        self.h = torch.randn(1, hidden)
        self.v = torch.randn(1, visible)
    def sample_h(self, x):
        # compute probability of hidden given visible
        # this is same as sigmoid activation function
        wx = torch.mm(x, self.weights)
        activation = wx + self.h.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        # compute probability of visible given hidden
        # this is same as sigmoid activation function
        wy = torch.mm(y, self.weights.t())
        activation = wy + self.v.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        # contrastive divergence using Gibbs sampling
        self.weights += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.v += torch.sum((v0 - vk), 0)
        self.h += torch.sum((ph0 - phk), 0)

# visible nodes    
nv = len(training[0])
# hidden nodes
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training
total_epoch = 10
for epoch in range(1, total_epoch+1):
    loss = 0
    s = 0.
    for id_user in range(0, max_users-batch_size, batch_size):
        vk = training[id_user:id_user+batch_size]
        v0 = training[id_user:id_user+batch_size]
        ph0, _ = rbm.sample_h(v0)
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)    
            vk[v0 < 0] = v0[v0 < 0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.
    print('epoch: ' + str(epoch))
    print('loss: ' + str(loss/s))
      
# Testing
test_loss = 0
s = 0.
for id_user in range(max_users):
    # we need the input of the training set to activate the neuron of rbm
    # to get the predicted rating of the test set
    v = training[id_user:id_user+1]
    vt = test[id_user:id_user+1]
    if len(vt[vt > 0]):
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)    
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
    s += 1.
print('test loss: ' + str(test_loss/s))
