"""
TODO:嵌入到算法类中
tensorboard --logdir=<path><to><target>

"""
import sys
import logbook
import numpy as np

from zipline.finance import commission
import os
from zipline.utils.tensorboard import TensorBoard

zipline_logging = logbook.NestedSetup([logbook.NullHandler(),
    logbook.StreamHandler(sys.stdout, level=logbook.INFO),
    logbook.StreamHandler(sys.stderr, level=logbook.ERROR),])

zipline_logging.push_application()

STOCKS = ['000001', '000333', '300017', '600000', '300001', '002024']
home_dir = os.path.expanduser('~')
log_path = os.path.join(home_dir, 'logdir')

# On-Line Portfolio Moving Average Reversion
# More info can be found in the corresponding paper:
# http://icml.cc/2012/papers/168.pdf
def initialize(context, eps=1,window_length=5):
    context.stocks = STOCKS
    context.sids = [context.symbol(stock) for stock in context.stocks]
    context.m = len(context.stocks)
    context.price = {}
    context.b_t = np.ones(context.m) / context.m
    context.last_desired_port = np.ones(context.m) / context.m
    context.eps = eps
    context.init = True
    context.days = 0
    context.window_length = window_length
    #context.add_transform('mavg', 5)

    context.set_commission(commission.PerShare(cost=0.005))

    sub_dir = 'eps{}'.format(eps)
    tb_log_dir = os.path.join(log_path, sub_dir)
    context.tensorboard = TensorBoard(log_dir=tb_log_dir)

def handle_data(context, data):
    context.days += 1
    if context.days < context.window_length:
        return

    if context.init:
        rebalance_portfolio(context, data, context.b_t)
        context.init = False
        return

    m = context.m

    x_tilde = np.zeros(m)
    #b = np.zeros(m)

    # find relative moving average price for each asset
    mavgs = data.history(context.sids, 'price', context.window_length, '1d').mean()
    for i, sid in enumerate(context.sids):
        price = data.current(sid, "price")
        # Relative mean deviation
        x_tilde[i] = mavgs[sid] / price

    ###########################
    # Inside of OLMAR (context 2)
    x_bar = x_tilde.mean()

    # market relative deviation
    mark_rel_dev = x_tilde - x_bar

    # Expected return with current portfolio
    exp_return = np.dot(context.b_t, x_tilde)
    weight = context.eps - exp_return
    variability = (np.linalg.norm(mark_rel_dev)) ** 2

    # test for divide-by-zero case
    if variability == 0.0:
        step_size = 0
    else:
        step_size = max(0, weight / variability)

    b = context.b_t + step_size * mark_rel_dev
    b_norm = simplex_projection(b)
    np.testing.assert_almost_equal(b_norm.sum(), 1)

    rebalance_portfolio(context, data, b_norm)

    # update portfolio
    context.b_t = b_norm

    # record something to show that these get logged
    # to tensorboard as well:
    context.record(x_bar=x_bar)

    if context.tensorboard is not None:
        # record context stats to tensorboard
        context.tensorboard.log_algo(context)


def rebalance_portfolio(context, data, desired_port):
    # rebalance portfolio
    desired_amount = np.zeros_like(desired_port)
    current_amount = np.zeros_like(desired_port)
    prices = np.zeros_like(desired_port)

    if context.init:
        positions_value = context.portfolio.starting_cash
    else:
        positions_value = context.portfolio.positions_value + \
            context.portfolio.cash

    for i, sid in enumerate(context.sids):
        current_amount[i] = context.portfolio.positions[sid].amount
        prices[i] = data.current(sid, "price")

    desired_amount = np.round(desired_port * positions_value / prices)

    context.last_desired_port = desired_port
    diff_amount = desired_amount - current_amount

    for i, sid in enumerate(context.sids):
        context.order(sid, diff_amount[i])


def simplex_projection(v, b=1):
    """Projection vectors to the simplex domain

    Implemented according to the paper: Efficient projections onto the
    l1-ball for learning in high dimensions, John Duchi, et al. ICML 2008.
    Implementation Time: 2011 June 17 by Bin@libin AT pmail.ntu.edu.sg
    Optimization Problem: min_{w}\| w - v \|_{2}^{2}
    s.t. sum_{i=1}^{m}=z, w_{i}\geq 0

    Input: A vector v \in R^{m}, and a scalar z > 0 (default=1)
    Output: Projection vector w

    :Example:
    >>> proj = simplex_projection([.4 ,.3, -.4, .5])
    >>> print(proj)
    array([ 0.33333333, 0.23333333, 0. , 0.43333333])
    >>> print(proj.sum())
    1.0

    Original matlab implementation: John Duchi (jduchi@cs.berkeley.edu)
    Python-port: Copyright 2013 by Thomas Wiecki (thomas.wiecki@gmail.com).
    """

    v = np.asarray(v)
    p = len(v)

    # Sort v into u in descending order
    v = (v > 0) * v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)

    rho = np.where(u > (sv - b) / np.arange(1, p + 1))[0][-1]
    theta = np.max([0, (sv[rho] - b) / (rho + 1)])
    w = (v - theta)
    w[w < 0] = 0
    return w


# Note: this function can be removed if running
# this algorithm on quantopian.com
def analyze(context=None, results=None):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    results.portfolio_value.plot(ax=ax)
    ax.set_ylabel('Portfolio value (USD)')
    plt.show()


## Note: this if-block should be removed if running
## this algorithm on quantopian.com
#if __name__ == '__main__':
#    import pandas as pd
#    from functools import partial
#    from zipline.algorithm import TradingAlgorithm
#    import os
#    home_dir = os.path.expanduser('~')
#    log_path = os.path.join(home_dir, 'logdir')

#    start = pd.Timestamp('2010-1-1',tz='utc')
#    end = pd.Timestamp('2017-10-21',tz='utc')
#    for eps in [1.0, 1.25, 1.5]:
#        algo_initialize = partial(initialize, eps = eps)
#        # Create and run the algorithm.
#        olmar = TradingAlgorithm(handle_data=handle_data,
#                                 initialize=initialize,
#                                 start=start,
#                                 end=end)

#        sub_dir = 'eps{}'.format(eps)
#        olmar.tb_log_dir = os.path.join(log_path, sub_dir)
        
#        print('-' * 100)
#        print(olmar.tb_log_dir)
#        results = olmar.run()

#        # Plot the portfolio data.
#        #analyze(results=results)
