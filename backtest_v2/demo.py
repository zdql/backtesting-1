from sympy import Q
from backtest import backtest
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf

'''
strat_function(preds, prices) - user specified mapping from past n days of price and analyst data to weights.
Returns: An array of asset weightings. The maximum weighting is 1, and the minimum is -1. The weights must sum to between -1 and 1.

Refer to test datasets for the shape of input data. Both preds and prices will be 2 dimensional arrays, with number of columns equal to number of assets + 1.
Number of days equal to number of rows. The first column will be date data.

Your strategy function needs to work with this data to geenrate portfolio weights.


'''

average_portfolio_allocations = {i: [] for i in range(9)}


class Strat:

    def rolling_view(self, lookback, num_assets):
        if len(self.price_data > lookback):
            returns = (
                self.price_data.iloc[-1] - self.price_data.iloc[-lookback]) / self.price_data.iloc[-lookback]
            indices = np.argsort(returns)

            num_indices = indices[:num_assets]
            return_weights = []
            for i in range(self.num_elems):
                if i in num_indices:
                    return_weights.append(1/num_assets)
                else:
                    return_weights.append(0)
            return return_weights
        else:
            return self.get_default_weights()

    def select_opt(self, num_assets):

        returns = self.ind_er

        var = self.ann_sd

        sharpe = returns / var

        indices = np.argsort(sharpe)

        num_indices = indices[:num_assets]

        if sum(num_indices) > 0:

            return_weights = []
            for i in range(self.num_elems):
                if i in num_indices:
                    return_weights.append(1/num_assets)
                else:
                    return_weights.append(0)
            return return_weights
        else:
            return self.get_default_weights()

    def __init__(self, price_data, analyst_data, outstanding_shares):
        self.num_elems = len(outstanding_shares)
        self.weights = {i: 0 for i in range(self.num_elems)}
        self.price_data = pd.DataFrame(price_data)
        self.analyst_data = pd.DataFrame(analyst_data)
        self.outstanding_shares = outstanding_shares

    def calculate_matrices(self, view=None):
        if view:
            df = self.price_data.iloc[-view:]

            self.cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()

            self.corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()

            self.ind_er = df.pct_change().mean()

            self.ann_sd = df.pct_change().std().apply(lambda x: x*np.sqrt(250))
        else:
            df = self.price_data
            df2 = self.analyst_data

            self.cov_matrix = (df.pct_change().apply(lambda x: np.log(
                1+x)).cov() + df2.pct_change().apply(lambda x: np.log(1+x)).cov()) / 2

            self.corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()

            self.ind_er = (df.pct_change().mean() +
                           df2.pct_change().mean()) / 2

            self.ann_sd = df.pct_change().std().apply(lambda x: x*np.sqrt(250))

    def get_default_weights(self):
        return list(self.weights.values())

    def update_weights(self, strat='opt', num=1, lookback=1):
        if len(self.price_data) > 2:

            if strat == 'rolling' and len(self.price_data) > max(lookback, num):
                return self.rolling_view(lookback, num)

            elif strat == 'opt':
                self.calculate_matrices(view=10)
                return self.select_opt(num)

            else:
                return self.get_default_weights()

        else:
            return self.get_default_weights()


def strat_function(preds, prices, last_weights):

    outstanding_shares = preds[0]
    price_data = preds[1:]

    strat = Strat(price_data, preds, outstanding_shares)
    # opt = strat.update_weights(strat='topxopt', num=1)
    opt = strat.update_weights(strat='opt', num=4)
    # opt = strat.update_weights(strat='ef', num=4)
    # opt = strat.update_weights(strat='rolling', num=4, lookback=10)
    # opt = strat.update_weights(strat='viewopt', num=100)
    # opt = strat.update_weights(strat='topadjxopt', num=4)

    for i in range(9):
        average_portfolio_allocations[i].append(opt[i])

    return opt


'''
Running the backtest - starting portfolio value of 10000, reading in data from these two locations.
'''
backtest(strat_function, 10000, '../test_datasets/Actual.csv',
         '../test_datasets/Predicted Testing Data Analyst 2.csv', True, "log.csv")

print("portfolio allocations: ")
for i in range(9):
    print(np.mean(average_portfolio_allocations[i]))
