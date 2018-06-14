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

# Architecture of AutoEncoder
# Inherited from Module class
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        # input is max_movies, hidden layer has 20 nodes
        self.fc1 = nn.Linear(max_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, max_movies)
        # Can use Rectifier or Sigmoid Activation function
        self.activation = nn.Sigmoid()
    def forward(self, x):
        # Encoding
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        # Decoding 
        x = self.fc4(x)
        return x
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training
nb_epoch = 200
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(max_users):
        # we create a batch of one input because pytorch/keras doesn't accept 
        # 1-D input so we add one more dimension
        input = Variable(training[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            # make sure that we don't compute the gradient with respect to target
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            # mean using just the movies that had non zero rating
            mean_corrector = max_movies/float(torch.sum(target.data>0) + 1e-10)
            # tells the direction in which we need to update the weight
            loss.backward()
            # loss is the squared error
            train_loss += np.sqrt(loss.data[0] * mean_corrector)
            s += 1.
            # tells the amount of weight that should be updated
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing
test_loss = 0
s = 0.
for id_user in range(max_users):
    # training set has all the movies but some ratings are missing for some users
    # we will predict that and test set has all the rating for all the movies
    # for the users which we will use to compare the results
    input = Variable(training[id_user]).unsqueeze(0)
    target = Variable(test[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = max_movies/float(torch.sum(target.data>0) + 1e-10)
        test_loss += np.sqrt(loss.data[0] * mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))