import pandas as pd

from zipline.pipeline.factors import SimpleMovingAverage
from zipline.pipeline import Fundamentals
from zipline.pipeline.data import USEquityPricing
from .factors import NDays


def IsStock():
    """
    股票过滤器

    Notes
    ------
        `USEquityPricing`包含股票指数数据，股票指数的tmv设置为0
    """
    return USEquityPricing.tmv.latest > 0.

def _common_filter(exclude_st):
    """通用过滤"""
    short_name = Fundamentals.short_name.latest
    common_stock = ~short_name.has_substring('退')
    not_st_stock = ~short_name.has_substring('ST')
    primary_share = IsStock()
    
    # Combine the above filters.
    if exclude_st:
        tradable_filter = (common_stock & primary_share & not_st_stock)
    else:
        tradable_filter = (common_stock & primary_share)
    return tradable_filter


def filter_universe(exclude_st = True):
    """

    Modified from Nathan Wolfe's Algorithm
    https://www.zipline.com/posts/pipeline-trading-universe-best-practice

    returns
    -------
    zipline.pipeline.Filter
        Filter for the set_screen of a Pipeline to create an
        ideal trading universe

    The filters:
        1. 正常交易的股票（不得包含`退`（退市）)
        2. 如过滤ST，则所有股票简称含有ST字样被过滤掉
        3. 存在市值（未退市）
        4. 移动平均成交额（前30%）
    """
    tradable_filter = _common_filter(exclude_st)
    # 简化点计算
    high_volume_tradable = SimpleMovingAverage(inputs=[USEquityPricing.amount],
                                               mask = tradable_filter,
                                               window_length=21).percentile_between(70, 100)

    screen = high_volume_tradable

    return screen


def CNN(N = 500, exclude_st = True):
    """
    成交额前N位过滤器

    参数
    _____
    N：整数
        取前N位
    exclude_st：布尔类型
        是否排除ST股票

    returns
    -------
    zipline.pipeline.Filter
        成交额前N位股票过滤器
    """

    tradable_filter = _common_filter(exclude_st)
    # 简化点计算
    high_volume_tradable = SimpleMovingAverage(inputs=[USEquityPricing.amount],
                                               mask = tradable_filter,
                                               window_length=21).top(N)
    screen = high_volume_tradable
    return screen


def SubNewStocks(days = 60):
    """
    次新股过滤器

    参数
    _____
    days：整数
        上市天数小于指定天数，判定为次新股

    returns
    -------
    zipline.pipeline.Filter
        次新股过滤器

    备注
    ----
        按自然日历计算上市天数
    """
    t_days = NDays()
    return t_days <= days