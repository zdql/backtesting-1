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
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

'''
strat_function(preds, prices) - user specified mapping from past n days of price and analyst data to weights.
Returns: An array of asset weightings. The maximum weighting is 1, and the minimum is -1. The weights must sum to between -1 and 1. 

Refer to test datasets for the shape of input data. Both preds and prices will be 2 dimensional arrays, with number of columns equal to number of assets + 1.
Number of days equal to number of rows. The first column will be date data.

Your strategy function needs to work with this data to geenrate portfolio weights.


'''

average_portfolio_allocations = {i: [] for i in range(40)}


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

    def efficient_frontier_weights(self):

        ef = EfficientFrontier(self.ind_er, self.cov_matrix)
        return list(ef.max_sharpe().values())

    def rolling_view(self, lookback, num_assets):
        if len(self.price_data > lookback):
            returns = (
                self.price_data.iloc[-1] - self.price_data.iloc[-lookback]) / self.price_data.iloc[-lookback]
            indices = np.argsort(returns)

            num_indices = indices[:len(num_assets)]
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

    def calculate_matrices(self):
        df = self.price_data

        self.cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()

        self.corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()

        self.ind_er = df.pct_change().mean()

        self.ann_sd = df.pct_change().apply(lambda x: np.log(
            1+x)).std().apply(lambda x: x*np.sqrt(250))

    def get_default_weights(self):
        return list(self.weights.values())

    def update_weights(self, strat='opt', num=1, lookback=1):
        if len(self.price_data) > 2:
            if strat == 'opt':
                self.calculate_matrices()
                self.opt = self.MaximizeSharpeRatioOptmzn(
                    self.ind_er.to_numpy(), self.cov_matrix.to_numpy(), 0.0, 40)
                return self.opt.x

            elif strat == 'topxopt':
                self.calculate_matrices()
                self.opt = self.MaximizeSharpeRatioOptmzn(
                    self.ind_er.to_numpy(), self.cov_matrix.to_numpy(), 0.0, 40)

                weights = self.opt.x
                ind = np.argpartition(weights, -num)[-num:]
                new_weights = []
                for i in range(self.num_elems):
                    if i in ind:
                        new_weights.append(weights[i])
                    else:
                        new_weights.append(0)

                normalized_weights = new_weights / np.linalg.norm(new_weights)

                return normalized_weights

            elif strat == 'ef':
                print(len(self.price_data))
                self.calculate_matrices()
                return self.efficient_frontier_weights()

            elif strat == 'rolling':
                return self.rolling_view(lookback, num)
            else:
                return self.get_default_weights()
        else:
            return self.get_default_weights()


def strat_function(preds, prices, last_weights):

    outstanding_shares = preds[0]
    price_data = preds[1:]

    strat = Strat(price_data, prices, outstanding_shares)
    # opt = strat.update_weights(strat='topxopt', num=4)
    opt = strat.update_weights(strat='rolling', num=5, lookback=10)
    for i in range(40):
        average_portfolio_allocations[i].append(opt[i])
    return opt


'''
Running the backtest - starting portfolio value of 10000, reading in data from these two locations.
'''
backtest(strat_function, 10000, '../test_datasets/price_data.csv',
         '../test_datasets/price_data.csv', True, "log.csv")

print("portfolio allocations: ")
for i in range(40):
    print(np.mean(average_portfolio_allocations[i]))
