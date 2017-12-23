"""
资产组合计算辅助函数
"""

import numpy as np
import pandas as pd


def last_trade(context, data, sid):
    """
    给定股票最近交易价格

    parameters
    ----------

    context:
        TradingAlgorithm context object
        passed to handle_data

    data : zipline.protocol.BarData
        data object passed to handle_data

    sid : zipline Asset
        the sid price

    """
    try:
        # # 签名更改
        return data.current(sid, 'price')
    except KeyError:
        return context.portfolio.positions[sid].last_sale_price



def get_current_holdings(context, data):
    """
    当前组合头寸。即股票持有的数量。

    parameters
    ----------

    context :
        TradingAlgorithm context object
        passed to handle_data

    data : zipline.protocol.BarData
        data object passed to handle_data

    returns
    -------
    pandas.Series
        contracts held of each asset

    """
    positions = context.portfolio.positions
    return pd.Series({stock: pos.amount
                      for stock, pos in positions.items()})



def get_current_allocations(context, data):
    """
    当前资产组合中各股票的权重

    w = 持有股票市值 / 资产组合市值

    parameters
    ----------

    context :
        TradingAlgorithm context object
        passed to handle_data

    data : zipline.protocol.BarData
        data object passed to handle_data

    returns
    -------
    pandas.Series
        current allocations as a percent of total liquidity

    notes
    -----
    资产组合市值 = 持有股票市值 + 现金

    """
    holdings = get_current_holdings(context, data)
    prices = pd.Series({sid: last_trade(context, data, sid)
                        for sid in holdings.index})
    return prices * holdings / context.portfolio.portfolio_value


