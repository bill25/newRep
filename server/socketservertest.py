

##########################################
#
#
#
# from MetaTrader5 import *
# from datetime import date
# import pandas as pd
# import matplotlib.pyplot as plt
# import MetaTrader5 as mt5
#
# # Initializing MT5 connection
# initialize()
#
#
#
#
# # Create currency watchlist for which correlation matrix is to be plotted
# sym = ['EURUSD','GBPUSD','USDJPY','USDCHF','AUDUSD','GBPJPY']
#
# # Copying data to dataframe
# d = pd.DataFrame()
# for i in sym:
#      rates = copy_rates_from_pos(i, mt5.TIMEFRAME_M1, 0, 1000)
#      d[i] = [y[4] for y in rates]
#
#
# # Deinitializing MT5 connection
# shutdown()
#
# # Compute Percentage Change
# rets = d.pct_change()
#
# # Compute Correlation
# corr = rets.corr()
#
# # Plot correlation matrix
# plt.figure(figsize=(10, 10))
# plt.imshow(corr, cmap='RdYlGn', interpolation='none', aspect='auto')
# plt.colorbar()
# plt.xticks(range(len(corr)), corr.columns, rotation='vertical')
# plt.yticks(range(len(corr)), corr.columns);
# plt.suptitle('FOREX Correlations Heat Map', fontsize=15, fontweight='bold')
# plt.show()
#
#
# # Importing statmodels for cointegration test
# import statsmodels
# from statsmodels.tsa.stattools import coint
#
# x = d['GBPUSD']
# y = d['GBPJPY']
# x = (x-min(x))/(max(x)-min(x))
# y = (y-min(y))/(max(y)-min(y))
#
# score = coint(x, y)
# print('t-statistic: ', score[0], ' p-value: ', score[1])
#
#
#
# # Plotting z-score transformation
# diff_series = (x - y)
# zscore = (diff_series - diff_series.mean()) / diff_series.std()
#
# plt.plot(zscore)
# plt.axhline(2.0, color='red', linestyle='--')
# plt.axhline(-2.0, color='green', linestyle='--')
#
# plt.show()









#################################################################################
#################################################################################
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

from keras import regularizers
from keras import optimizers
class train:


    def process(self,candles, type,volume):
        """
        processing candles to a format/shape consumable for the model
        :param candles: dict/list of Open, High, Low, Close prices
        :return: X: numpy.ndarray, Y: numpy.ndarray
        """

        decimal_figures = 6
        y_change_threshold = 0.00018
        if type=='train':
            if volume == 1:
                X = np.ndarray(shape=(0, 5))
            if volume == 0:
                X = np.ndarray(shape=(0, 4))
            Y = np.ndarray(shape=(0, 1))

            # clean and process data
            previous_close = None

            for candle in candles:
                if volume==1:
                    X = np.append(X,
                                  np.array([[
                                      # High 2 Open Price
                                      round(candle['high'] / candle['open'] - 1, decimal_figures),
                                      # Low 2 Open Price
                                      round(candle['low'] / candle['open'] - 1, decimal_figures),
                                      # Close 2 Open Price
                                      round(candle['close'] / candle['open'] - 1, decimal_figures),
                                      # High 2 Low Price
                                      round(candle['high'] / candle['low'] - 1, decimal_figures),

                                      round(1/candle['tick_volume'], decimal_figures)]]),
                                  axis=0)
                else:
                      X = np.append(X,
                                    np.array([[
                                        # High 2 Open Price
                                        round(candle['high'] / candle['open'] - 1, decimal_figures),
                                        # Low 2 Open Price
                                        round(candle['low'] / candle['open'] - 1, decimal_figures),
                                        # Close 2 Open Price
                                        round(candle['close'] / candle['open'] - 1, decimal_figures),
                                        # High 2 Low Price
                                        round(candle['high'] / candle['low'] - 1, decimal_figures)]]),


                                    axis=0)


                # Compute the Y / Target Variable
                if previous_close is not None:
                    y = 0
                    precise_prediction = round(1 - previous_close / candle['close'], decimal_figures)

                    # positive price change more growth than threshold
                    if precise_prediction > y_change_threshold:
                        y = 1
                    # negative price change with more decline than threshold
                    elif precise_prediction < 0 - y_change_threshold:
                        y = 2
                    # price change in between positive and negative threshold
                    elif precise_prediction < y_change_threshold and precise_prediction > 0 - y_change_threshold:
                        y = 0

                    Y = np.append(Y, np.array([[y]]))
                else:
                    Y = np.append(Y, np.array([[0]]))
                previous_close = candle['close']

            Y = np.delete(Y, 0)
            Y = np.append(Y, np.array([0]))
            Y = to_categorical(Y, num_classes=3)


            return X, Y


    def get_lstm_model(self,lrs):
        model = Sequential()
        model.add(LSTM(units=20, input_shape=(5,), return_sequences=True))
        model.add(LSTM(units=20))
        model.add(Dense(units=3,
                                activation='softmax'))
        sgd = optimizers.SGD(lr=lrs)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        return model

    def get_model(self,layer,volume,lrs):
        """
        Here we define our model Layers using Keras
        :return: Keras Model Object
        """
        inputsize = 4
        if volume==1:
            inputsize=5
        model = Sequential()
        if(layer==2):
                model.add(Dense(units=16,
                                activation='relu',
                                input_shape=(inputsize,)))
                model.add(Dense(units=16,
                                activation='relu',
                                kernel_regularizer=regularizers.l2(0.001),
                                activity_regularizer=regularizers.l1(0.001)))
                model.add(Dense(units=3,
                                activation='softmax'))
        elif(layer==3):
            model.add(Dense(units=16,
                            activation='relu',
                            input_shape=(inputsize,)))
            model.add(Dense(units=16,
                            activation='relu',
                            kernel_regularizer=regularizers.l2(0.001),
                            activity_regularizer=regularizers.l1(0.001)))
            model.add(Dense(units=16,
                            activation='relu',
                            kernel_regularizer=regularizers.l2(0.001),
                            activity_regularizer=regularizers.l1(0.001)))

            model.add(Dense(units=3,
                            activation='softmax'))



        sgd = optimizers.SGD(lr=lrs)

        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        return model

    week = 60 * 60 * 24 * 7
    decimal_figures = 6
    y_change_threshold = 0.001

    def predict(self):
        # """
        # :return: Read Data from MongoDB. Apply ML Model on Data
        # """
        #
        # # read data from MongoDB
        # collection = db['streaming_data']
        # newest_candle = collection.find({}) \
        #     .sort([('candles.0.time', -1)]) \
        #     .limit(1)
        #
        # for nc in newest_candle:
        #     newest_candle = nc['candles'][0]['mid']
        #
        # # get processed X
        # X = np.ndarray(shape=(0, 4))
        #
        #
        #
        # decimal_figures = 6
        # X = np.append(X,
        #               np.array([[
        #                   # High 2 Open Price
        #                   round(newest_candle['h'] / newest_candle['o'] - 1, decimal_figures),
        #                   # Low 2 Open Price
        #                   round(newest_candle['l'] / newest_candle['o'] - 1, decimal_figures),
        #                   # Close 2 Open Price
        #                   round(newest_candle['c'] / newest_candle['o'] - 1, decimal_figures),
        #                   # High 2 Low Price
        #                   round(newest_candle['h'] / newest_candle['l'] - 1, decimal_figures)]]),
        #               axis=0)
        #
        # print(X)

        last = load_model('2020-03-23-16-33-55-EUR_USD_H1')





    def retrain(self,candles,epochs,layer,batch_size,volume,lr):
        """
        Retrains a model for a specific a) trading instrument, b) timeframe, c) input shape
        """

        # get historical data from data service

        X, Y = self.process(candles,"train",volume)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

        model =self.get_lstm_model(lr)
        fit = model.fit(X_train, Y_train, epochs=epochs, verbose=True)
        score = model.evaluate(X_test, Y_test, batch_size=batch_size)
        print("this is score  =================> :",score)
        print(model.summary())

        # TODO: Save trained model to disk
        filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model.save("lr "+str(lr)+" volume "+str(volume) + " batch size "+str(batch_size)+" epochs "+str(epochs) +"  layers "+ str(layer) + " towLayer  "+ str(score[0]) + "  " + str(score[1]) )






#################################################################################
#################################################################################




# -*- coding: utf-8 -*-
import MetaTrader5 as mt5
"""
Created on Thu Mar 14 16:13:03 2019

@author: dmitrievsky
"""
import MetaTrader5 as mt5
from MetaTrader5 import *
from datetime import datetime
import pandas as pd
# Initializing MT5 connection
initialize()



# Copying data to pandas data frame
import matplotlib.pyplot as plt

rates = copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M10, 0, 600)
# Deinitializing MT5 connection
shutdown()


# x,y=train.process(rates,"train",0)
# plt.hist(y)
# plt.show()
# stockdata = pd.DataFrame(rates)
# stockdata['time']=pd.to_datetime(rates['time'], unit='s')
# print(stockdata)

# print(stockdata.tick_volume)


# for col in stockdata.columns:
#     print(col)
# train().retrain(candles=rates)
# pour extraire le bon model qui predit le mieux


epochs=[1000]
batch_sizes=[32,64,128]
learning_ratess=[0.0001,0.001]
layers=[2]
volume=[1]


for epoch in epochs:
    for batch_size in batch_sizes:
        for layer in layers:
            for v in volume:
                for lr in learning_ratess:
                     train().retrain(rates,epoch,layer,batch_size,v,lr)


# PREDICTION OF THE NEWST CANDLE
# initialize()
# model = load_model('volume 1 batch size 64 epochs 1000  layers 2 towLayer  0.3721009327901734  0.9105555415153503')
# for i in range(50000):
#         newest_candle = copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M10, i, 1)
#         x,y=train.process(newest_candle,"train",1)
#         Y = model.predict(x)
#         if Y[0][1] > 0.5:
#             print("prdicted and position ",Y,"  ",i)







#
# import plotly.graph_objs as go
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#
# trace = go.Ohlc(x=stockdata['time'],
#                 open=stockdata['open'],
#                 high=stockdata['high'],
#                 low=stockdata['low'],
#                 close=stockdata['close'])
#
# data = [trace]
# plot(data)


