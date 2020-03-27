import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import scipy.optimize as solver
import datetime as dt
from functools import reduce
import MetaTrader5 as mt5
from MetaTrader5 import *
from datetime import datetime

class price_extractor:

    def __init__(self, api, companies):
        print('Initialised Price Extractor')
        self.__api = api
        self.__companies = companies
        print("hello :",companies)

        pass

    def get_prices(self,  event, start_date, end_date):

        prices = pd.DataFrame()
        symbols = self.__companies['Ticker']
        tmp={}
        for i in symbols:
            # try:
            initialize()
            tmp = copy_rates_from_pos(i, mt5.TIMEFRAME_M5, 0, 100)
            stockdata = pd.DataFrame(tmp)
            stockdata['time'] = pd.to_datetime(tmp['time'], unit='s')
            stockdata.rename({'time': 'Date'}, axis=1, inplace=True)
            stockdata.set_index('Date', inplace=True)
            shutdown()
            # tmp = web.DataReader(i, self.__api, start_date, end_date)
            # print('Fetched prices for: '+i)
            # print(tmp.head())
            #
            # for col in tmp.columns:
            #     print(col)

            # except:
            #     print('Issue getting prices for: '+i)
            # else:
            prices[i] = tmp[event]

        return prices