"""
区别于股票代码，指数代码请添加"I"，如000300(沪深300)，其符号I000300
"""

import pandas as pd

from cswd.utils import ensure_list, sanitize_dates
#from cswd.sqldata.stock_daily import StockDailyData
from cswd.sqldata.query import query_adjusted_pricing
from cswd.sqldata.stock_index_daily import StockIndexDailyData
from zipline.assets._assets import Equity

index_sid_to_code = lambda x:str(int(x) - 100000).zfill(6)  # 指数sid -> 指数代码


def _pricing_factory(code, fields, start, end, normalize):
    is_index = False
    if isinstance(code, str):
        symbol = code
    elif isinstance(code, Equity):
        symbol = code.symbol
    else:
        raise TypeError('code只能是str或Equity类型')
    if len(symbol) == 7:
        is_index = True
        symbol = symbol[1:]
    if symbol[0] in ('1','4'):
        is_index = True
        symbol = index_sid_to_code(symbol)
    if is_index:
        df = StockIndexDailyData.query_by_code(symbol).loc[start:end, :]
        df = df.reindex(columns=fields, fill_value=0)
    else:
        #df = StockDailyData.query_by_code(symbol).loc[start:end, fields]
        df = query_adjusted_pricing(symbol, start, end, fields, normalize)
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    return df


def get_pricing(codes, fields = 'close', start = None, end = None, 
                assets_as_columns = False,
                normalize = False):
    """
    加载股票或指数期间日线数据

    Parameters
    ----------
    codes : [str或Equity对象]
        股票或指数代码列表，或Equity对象
    fields : [str]
        读取列名称
    start : datetime-like
        开始日期，默认 today - 365
    end : datetime-like
        结束日期，默认 today
    assets_as_columns : bool
        是否将符号作为列名称。默认为假。
    normalize : bool
        是否将首日价格调整为1.0。默认为假。

    Examples
    --------
    >>> get_pricing(['I000001','000002','000333']).tail()
                            close
    date       code              
    2017-12-07 000333     49.1300
               I000001  3272.0542
    2017-12-08 000002     29.8200
               000333     50.4300
               I000001  3289.9924
    >>> # 指数数据中不存在的字段以0值填充
    >>> get_pricing(['I000001','000002','000333'], ['high','low', 'cmv']).tail()
                             high        low           cmv
    date       code                                       
    2017-12-07 000333     51.0800    49.0500  3.165395e+11
               I000001  3291.2817  3259.1637  0.000000e+00
    2017-12-08 000002     29.8800    29.0500  2.899755e+11
               000333     51.2000    49.4000  3.249153e+11
               I000001  3297.1304  3258.7593  0.000000e+00
    >>> # 股票代码作为列名称
    >>> get_pricing(['I000001','000002','000333'], 'close', assets_as_columns=True).tail()
    code        000002  000333    I000001
    date                                 
    2017-12-04   30.85   51.50  3309.6183
    2017-12-05   31.03   52.18  3303.6751
    2017-12-06   30.77   50.95  3293.9648
    2017-12-07   29.95   49.13  3272.0542
    2017-12-08   29.82   50.43  3289.9924
    >>> get_pricing(['I000001','000002','000333'], 'close', assets_as_columns=True, normalize=True).tail()
    code          000002    000333    I000001
    date                                     
    2017-12-11  1.466743  1.912764  3322.1956
    2017-12-12  1.449101  1.901349  3280.8136
    2017-12-13  1.467234  1.958425  3303.0373
    2017-12-14  1.475564  1.944156  3292.4385
    2017-12-15  1.426558  1.915618  3266.1371
    """
    if start is None:
        start = pd.Timestamp('today') - pd.Timedelta(days=365)
    if end is None:
        end = pd.Timestamp('today')
    start, end = sanitize_dates(start, end)
    codes = ensure_list(codes)
    fields = ensure_list(fields)
    dfs = []
    for code in codes:
        try:
            # 可能期间不存在数据
            df = _pricing_factory(code, fields, start, end, normalize)
        except:
            df = pd.DataFrame()
        df['code'] = code
        dfs.append(df)
    res = pd.concat(dfs)
    res.set_index('code', append=True, inplace=True)
    res.sort_index(level=0, inplace=True)
    if assets_as_columns:
        if len(fields) != 1:
            msg = '当符号调整为列时，字段长度应为1，输入字段长度为{}'
            raise ValueError(msg.format(len(fields)))
        res = res[fields].unstack()
        res.columns = res.columns.get_level_values(1)
    return res