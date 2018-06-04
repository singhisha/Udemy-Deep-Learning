# -*- coding: utf-8 -*-
# Recurrent Neural Network

# Part - 1 Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# feature scaling
# Normalization preferred for rnn over standardization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# creating a dataset with 60 time steps and 1 output
# so we take data from 60 previous financial days before time t to try to predict
# for time period t+1
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
    
# reshaping
# nb of indicators/predictors, here we just have the google stock prices
# but we might have some other companies stock prices affecting our stock
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part - 2 Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

# initialising the RNN
regressor = Sequential()

# Adding LSTM layer and Dropout
# we need high dimensionality so we choose 50 units
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(p = 0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(p = 0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(p = 0.2))

# return_sequences is False because it is the last LSTM layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(p = 0.2))
 
# adding the output layer
# since we are predicting only one value so units is 1
regressor.add(Dense(units = 1))

# compile the rnn
# RMSprop is preferred for rnn but Adam is always a safe and powerful choice
# MSE as loss function for regression problem
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# fitting the rnn to training set

regressor.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Making the predictions and visualisation
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock prices
# we should never change the actual test values
# since we need the previous 60 observations some of the values come from the training data
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 : ].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
# inverse the scaled values to get the real values
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualising the final result
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time: Jan 3rd - Jan 31st')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

#Parameter Tuning for Regression is the same as Parameter Tuning for 
#Classification for Artificial Neural Networks, 
#the only difference is that we have to replace:
#scoring = 'accuracy'  
#by:
#scoring = 'neg_mean_squared_error'