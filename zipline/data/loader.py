#
# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""
该模块原设计是从网络读取数据，存储在本地文件。
现更改为直接从本地sql数据库中读取，去掉相应辅助函数。
"""
import os

import logbook
import pandas as pd

# # 读取本地指数作为基准收益率
#from .benchmarks import get_benchmark_returns
from .benchmarks_cn import get_benchmark_returns
# # 
from . import treasuries, treasuries_can, treasuries_cn
from zipline.utils.calendars import get_calendar


logger = logbook.Logger('Loader')


# 所有指数均对应treasuries_cn
# # Mapping from index symbol to appropriate bond data
INDEX_MAPPING = {
    '000300':
    (treasuries_cn,'treasury_curves_cn.csv',''),
    'SPY':
    (treasuries, 'treasury_curves.csv', 'www.federalreserve.gov'),
    '^GSPTSE':
    (treasuries_can, 'treasury_curves_can.csv', 'bankofcanada.ca'),
    '^FTSE':  # use US treasuries until UK bonds implemented
    (treasuries, 'treasury_curves.csv', 'www.federalreserve.gov'),
}

# 测试使用
def get_benchmark_filename(symbol):
    return "%s_benchmark.csv" % symbol

def load_market_data(trading_day=None, trading_days=None, bm_symbol='000300',
                     environ=None):
    """
    Load benchmark returns and treasury yield curves for the given calendar and
    benchmark symbol.

    Benchmarks are downloaded as a Series from Google Finance.  Treasury curves
    are US Treasury Bond rates and are downloaded from 'www.federalreserve.gov'
    by default.  For Canadian exchanges, a loader for Canadian bonds from the
    Bank of Canada is also available.

    Results downloaded from the internet are cached in
    ~/.zipline/data. Subsequent loads will attempt to read from the cached
    files before falling back to redownload.

    Parameters
    ----------
    trading_day : pandas.CustomBusinessDay, optional
        A trading_day used to determine the latest day for which we
        expect to have data.  Defaults to an NYSE trading day.
    trading_days : pd.DatetimeIndex, optional
        A calendar of trading days.  Also used for determining what cached
        dates we should expect to have cached. Defaults to the NYSE calendar.
    bm_symbol : str, optional
        Symbol for the benchmark index to load.  Defaults to 'SPY', the Google
        ticker for the S&P 500.

    Returns
    -------
    (benchmark_returns, treasury_curves) : (pd.Series, pd.DataFrame)

    Notes
    -----

    Both return values are DatetimeIndexed with values dated to midnight in UTC
    of each stored date.  The columns of `treasury_curves` are:

    '1month', '3month', '6month',
    '1year','2year','3year','5year','7year','10year','20year','30year'
    """
    calendar = get_calendar('SZSH')
    #if trading_day is None:
    #    trading_day = get_calendar().trading_day
    #if trading_days is None:
    #    trading_days = get_calendar().all_sessions
    if trading_day is None:
        # 更改为calendar.day
        trading_day = calendar.day
    if trading_days is None:
        trading_days = calendar.all_sessions

    first_date = trading_days[0]
    now = pd.Timestamp.utcnow()

    # We expect to have benchmark and treasury data that's current up until
    # **two** full trading days prior to the most recently completed trading
    # day.
    # Example:
    # On Thu Oct 22 2015, the previous completed trading day is Wed Oct 21.
    # However, data for Oct 21 doesn't become available until the early morning
    # hours of Oct 22.  This means that there are times on the 22nd at which we
    # cannot reasonably expect to have data for the 21st available.  To be
    # conservative, we instead expect that at any time on the 22nd, we can
    # download data for Tuesday the 20th, which is two full trading days prior
    # to the date on which we're running a test.

    # We'll attempt to download new data if the latest entry in our cache is
    # before this date.

    #last_date = trading_days[trading_days.get_loc(now, method='ffill') - 2]

    # # 根据实际情况调整偏移天数
    local_now = pd.Timestamp('now')
    offset = 1
    refresh_time = local_now.normalize().replace(hour=18)
    actual_end = calendar.actual_last_session
    if local_now.date() > actual_end.date():
        offset = 0
    elif local_now > refresh_time:
        offset = 0

    last_date = trading_days[trading_days.get_loc(now, method='ffill') - offset]

    br = get_benchmark_returns(bm_symbol, first_date, last_date)

    tc = treasuries_cn.get_treasury_data(first_date, last_date)

    benchmark_returns = br[br.index.slice_indexer(first_date, last_date)]
    treasury_curves = tc[tc.index.slice_indexer(first_date, last_date)]

    msg_fmt = 'Read benchmark and treasury data for {} from {} to {}'

    logger.info(
        msg_fmt.format(bm_symbol, 
                       (first_date - trading_day).strftime('%Y-%m-%d'),
                       last_date.strftime('%Y-%m-%d'))
    )
    return benchmark_returns, treasury_curves


def load_prices_from_csv(filepath, identifier_col, tz='UTC'):
    data = pd.read_csv(filepath, index_col=identifier_col)
    data.index = pd.DatetimeIndex(data.index, tz=tz)
    data.sort_index(inplace=True)
    return data


def load_prices_from_csv_folder(folderpath, identifier_col, tz='UTC'):
    data = None
    for file in os.listdir(folderpath):
        if '.csv' not in file:
            continue
        raw = load_prices_from_csv(os.path.join(folderpath, file),
                                   identifier_col, tz)
        if data is None:
            data = raw
        else:
            data = pd.concat([data, raw], axis=1)
    return data
