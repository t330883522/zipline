import pandas as pd

from cswd.websource.utils import ensure_list, sanitize_dates
from cswd.sqldata.stock_daily import StockDailyData
from cswd.sqldata.stock_index_daily import StockIndexDailyData

def _get_pricing(read_func, codes, fields, start, end):
    if start is None:
        start = pd.Timestamp('today') - pd.Timedelta(days=365)
    if end is None:
        end = pd.Timestamp('today')
    start, end = sanitize_dates(start, end)
    codes = ensure_list(codes)
    fields = ensure_list(fields)
    dfs = []
    for code in codes:
        df = read_func(code)
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df)
        df.loc[:,'code'] = code
        dfs.append(df)
    res = pd.concat(dfs)
    #res.set_index('code', append=True, inplace=True)
    res.reset_index(inplace=True)
    return res

def get_pricing(stocks, fields = 'close', start=None, end=None):
    """
    加载股票期间日线收盘价

    Parameters
    ----------
    stocks : [str]
        股票代码列表
    fields : [str]
        读取列名称
    start : datetime-like
        开始日期，默认 today - 365
    end : datetime-like
        结束日期，默认 today

    Examples
    --------
    >>> get_pricing(['000001','000002','000333']).tail()
                date  close    code
    14143 2017-11-27  52.64  000333
    14144 2017-11-28  52.59  000333
    14145 2017-11-29  51.10  000333
    14146 2017-11-30  51.18  000333
    14147 2017-12-01  50.58  000333
    >>> get_pricing(['000001','000002','000333'], ['high','low']).tail()
                date   high    low    code
    14143 2017-11-27  53.40  51.75  000333
    14144 2017-11-28  53.07  51.78  000333
    14145 2017-11-29  52.77  50.20  000333
    14146 2017-11-30  52.06  50.61  000333
    14147 2017-12-01  51.67  50.50  000333
    """
    func = lambda x:StockDailyData.query_by_code(x).loc[start:end, fields]
    res = _get_pricing(func, stocks, fields, start, end)
    return res


def get_index_pricing(index_codes, fields = 'close', start=None, end=None):
    """
    加载指数期间日线收盘价

    Parameters
    ----------
    index_codes : [str]
        指数代码列表
    fields : [str]
        读取列名称
    start : datetime-like
        开始日期，默认 today - 365
    end : datetime-like
        结束日期，默认 today

    Examples
    --------
    >>> get_index_pricing(['000001','000300']).tail()
                date      close    code
    10448 2017-11-27  4049.9475  000300
    10449 2017-11-28  4055.8235  000300
    10450 2017-11-29  4053.7529  000300
    10451 2017-11-30  4006.0993  000300
    10452 2017-12-01  3998.1365  000300
    >>> get_index_pricing(['000001','000300'], ['high','low']).tail()
                date       high        low    code
    10448 2017-11-27  4088.9318  4037.1457  000300
    10449 2017-11-28  4056.4753  4010.2955  000300
    10450 2017-11-29  4069.2418  4004.6870  000300
    10451 2017-11-30  4052.7970  3988.0018  000300
    10452 2017-12-01  4027.1064  3984.3776  000300
    """
    func = lambda x:StockIndexDailyData.query_by_code(x).loc[start:end, fields]
    res = _get_pricing(func, index_codes, fields, start, end)
    return res
