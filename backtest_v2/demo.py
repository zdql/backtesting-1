from backtest import backtest
import numpy as np
from scipy.optimize import minimize
import pandas_datareader.data as reader
import pandas as pd
import datetime as dt
import statsmodels.api as sm
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy import optimize

'''
strat_function(preds, prices) - user specified mapping from past n days of price and analyst data to weights.
Returns: An array of asset weightings. The maximum weighting is 1, and the minimum is -1. The weights must sum to between -1 and 1. 

Refer to test datasets for the shape of input data. Both preds and prices will be 2 dimensional arrays, with number of columns equal to number of assets + 1.
Number of days equal to number of rows. The first column will be date data.

Your strategy function needs to work with this data to geenrate portfolio weights.


'''


class Strat:

    def MaximizeSharpeRatioOptmzn(self, MeanReturns, CovarReturns, RiskFreeRate, PortfolioSize):

        # define maximization of Sharpe Ratio using principle of duality
        def f(x, MeanReturns, CovarReturns, RiskFreeRate, PortfolioSize):
            funcDenomr = np.sqrt(np.matmul(np.matmul(x, CovarReturns), x.T))
            funcNumer = np.matmul(np.array(MeanReturns), x.T)-RiskFreeRate
            func = -(funcNumer / funcDenomr)
            return func

        # define equality constraint representing fully invested portfolio
        def constraintEq(x):
            A = np.ones(x.shape)
            b = 1
            constraintVal = np.matmul(A, x.T)-b
            return constraintVal

        # define bounds and other parameters
        xinit = np.repeat(0.33, PortfolioSize)
        cons = ({'type': 'eq', 'fun': constraintEq})
        lb = 0
        ub = 1
        bnds = tuple([(lb, ub) for x in xinit])

        # invoke minimize solver
        opt = optimize.minimize(f, x0=xinit, args=(MeanReturns, CovarReturns,
                                                   RiskFreeRate, PortfolioSize), method='SLSQP',
                                bounds=bnds, constraints=cons, tol=10**-3)

        return opt

    def __init__(self, price_data, analyst_data, outstanding_shares):
        self.num_elems = len(outstanding_shares)
        self.weights = {i: 0 for i in range(self.num_elems)}
        self.price_data = pd.DataFrame(price_data)
        self.analyst_data = pd.DataFrame(analyst_data)
        self.outstanding_shares = outstanding_shares

    def calculate_matrices(self):
        df = self.price_data

        self.cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()

        self.corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()

        self.ind_er = df.pct_change().mean()

        self.ann_sd = df.pct_change().apply(lambda x: np.log(
            1+x)).std().apply(lambda x: x*np.sqrt(250))

    def get_default_weights(self):
        return list(self.weights.values())

    def update_weights(self):
        if len(self.price_data) > 1:
            self.calculate_matrices()
            self.opt = self.MaximizeSharpeRatioOptmzn(
                self.ind_er.to_numpy(), self.cov_matrix.to_numpy(), 0.0, 40)
            return self.opt.x
        else:
            return self.get_default_weights()


def strat_function(preds, prices, last_weights):

    outstanding_shares = preds[0]
    price_data = preds[1:]

    strat = Strat(price_data, prices, outstanding_shares)
    opt = strat.update_weights()
    return opt


'''
Running the backtest - starting portfolio value of 10000, reading in data from these two locations.
'''
backtest(strat_function, 10000, '../test_datasets/price_data.csv',
         '../test_datasets/price_data.csv', True, "log.csv")
