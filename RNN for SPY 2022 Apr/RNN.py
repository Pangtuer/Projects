#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 09:41:23 2022

@author: niannianliu
"""
# Recurrent Neural Network

# Part 1 - Data Preprocessing
# Importing the librariers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import training set
dataset_train = pd.read_csv('SPY_Train.csv')
training_set = dataset_train.iloc[:, 4:5].values  # Use Close Price

# Feature scaling
from sklearn.preprocessing import MinMaxScaler  
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set) 

# Create a data structure with 60 timesteps and 1 output
X_train = []
y_train = []


for i in range(60, 1180):
    X_train.append(training_set_scaled[i-60: i, 0]) # 0 to 60 but 60 is excluded so X_train only to 59
    y_train.append(training_set_scaled[i, 0])   # y_train starts from 60
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building RNN
# Importing Kearas liabraries and packages
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Initialising RNN
regressor = Sequential()  

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))  
regressor.add(Dropout(0.2))
   
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True)) 
regressor.add(Dropout(0.2))  

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True)) 
regressor.add(Dropout(0.2)) 

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2)) 

# Adding the output layer
regressor.add(Dense(units = 1)) 

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') # Optimizer can also use RMSProp

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)  

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price
dataset_test = pd.read_csv('SPY_Test.csv')
real_stock_price = dataset_test.iloc[:, 4:5].values  # Use Close Price

# Getting the predicted stock price from original dataframe
dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis = 0)  

inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60: ].values

# Only change inputs scale, not test values
inputs = inputs.reshape(-1, 1) 
inputs = sc.transform(inputs) 
X_test = []
for i in range(60, 139): # prior 60 periods + current 79 periods = 139
     X_test.append(inputs[i-60: i, 0])    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real SPY Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted SPY Price')
plt.title('SPY Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('SPY Price')
plt.legend()
plt.show()