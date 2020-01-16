'''
Made by Mujtaba Siddiqui, January 2020

Description: Using a recurrent neural network known as Long Short Term Memory (LSTM) this program can predict the
             closing stock price of a corporation using the previous 60 days stock prices.

Dependancies: Python 3, Tensorflow, Keras, Pandas, Numpy, Scikit-Learn, Matplotlib
'''

#IMPORTS
import math
import pandas_datareader as dr
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#Plot Style
plt.style.use('ggplot')

def predictor (stock):
    #Get stock price
    df = dr.DataReader(stock, data_source='yahoo', start='2011-05-26', end='2020-01-01')

    #df with just Close Price
    close = df.filter(['Close'])

    #Make close a numpy array
    dataset = close.values

    #Use 80% of data to train model
    trainData_len = math.ceil(len(dataset) * 0.8)

    #Scale data
    scale = MinMaxScaler(feature_range=(0,1))

    scaled = scale.fit_transform(dataset)

    #Create train dataset
    trainData = scaled[0:trainData_len, :]

    #Split training data
    Xtrain = []
    Ytrain = []

    for i in range(60, len(trainData)):
        Xtrain.append(trainData[i-60:i, 0])
        Ytrain.append(trainData[i,0])

    #Convert split training data to numpy array
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    #Reshape data
    Xtrain = np.reshape(Xtrain,(Xtrain.shape[0], Xtrain.shape[1], 1))

    #Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(Xtrain.shape[1],1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    #Compile Model
    model.compile(optimizer='adam', loss="mean_squared_error")

    #Train Model
    model.fit(Xtrain, Ytrain, batch_size=1, epochs=1)

    #Create testing data
    test = scaled[trainData_len-60: len(dataset), :]

    #XY Test Data
    Xtest = []
    Ytest = dataset[trainData_len: , :]

    for i in range(60, len(test)):
        Xtest.append(test[i-60:i, 0])

    #Convert data to numpy array
    Xtest = np.array(Xtest)

    #Reshape data
    print(Xtest.shape)
    Xtest = np.reshape(Xtest,(Xtest.shape[0], Xtest.shape[1], 1))

    #Get model to predict price
    prediction = model.predict(Xtest)
    prediction = scale.inverse_transform(prediction)

    #Check Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean(prediction- Ytest) ** 2)

    print(rmse)

    #Plot the data
    train = close[:trainData_len]
    valid = close[trainData_len:]
    valid['Predictions'] = prediction

    #Visualize
    plt.figure(figsize=(16,8))
    plt.title('Close Price Prediction')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.xlabel('Date', fontsize= 18)
    plt.ylabel('Close Price $USD', fontsize=18)
    plt.legend(['Train', 'Val', 'Predictions'])
    plt.show()

company = input('What stock would you like to predict (please use the short form): ')

predictor(company)