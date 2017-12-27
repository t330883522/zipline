from functools import lru_cache
import pandas as pd
from odo import odo

from zipline.pipeline.fundamentals.base import STOCK_DB
from cswd.websource.utils import MARKET_START
from cswd.websource.utils import ensure_list
from cswd.sqldata.core import fetch_single_stock_equity

def symbols_to_sids(symbols):
    """股票代码转换为sid"""
    symbols = ensure_list(symbols)
    res = [int(x) for x in symbols]
    if len(res) == 1:
        return res[0]
    else:
        return res

def sids_to_symbols(sids):
    """sid转换为股票代码"""
    sids = ensure_list(sids)
    res = [str(x).zfill(6) for x in set(sids)]
    if len(res) == 1:
        return res[0]
    else:
        return res

def nanos_to_seconds(nanos):
    return nanos / (1000 * 1000 * 1000)

@lru_cache()
def _load_raw_data_from_sql(asset_id):
    """从本地数据库读取股票日线交易所有原始数据"""
    symbol = sids_to_symbols(asset_id)
    frame = fetch_single_stock_equity(symbol, 
                                start_date=MARKET_START.date(),
                                end_date=pd.Timestamp('today').date())
    frame['day'] = nanos_to_seconds(frame.index.asi8)
    frame['id'] = asset_id
    return frame

def make_bar_data(asset_info, calendar):
    """
    对于给的的组合（asset/date/column）生成要写入的原始数据

    Parameters
    ----------
    asset_info : DataFrame
        DataFrame with asset_id as index and 'start_date'/'end_date' columns.
    calendar : pd.DatetimeIndex
        The trading calendar to use.

    Yields
    ------
    p : (int, pd.DataFrame)
        A sid, data pair to be passed to BcolzDailyDailyBarWriter.write
    """

    assert (asset_info['start_date'] < asset_info['end_date']).all()

    def _raw_data_for_asset(asset_id):
        """
        Generate 'raw' data that encodes information about the asset.

        See docstring for a description of the data format.
        """
        # Get the dates for which this asset existed according to our asset
        # info.
        symbol = sids_to_symbols(asset_id)
        datetimes = calendar[calendar.slice_indexer(
            asset_start(asset_info, asset_id),
            asset_end(asset_info, asset_id),
        )]
        # # 数据写入时已经处理好dtype，无需再进行限定
        frame = fetch_single_stock_equity(symbol).loc[datetimes[0]:datetimes[-1], :]
        return frame

    for asset in asset_info.index:
        yield asset, _raw_data_for_asset(asset)

def expected_bar_value(asset_id, date, colname):
    """
    Check that the raw value for an asset/date/column triple is as
    expected.

    Used by tests to verify data written by a writer.
    """
    frame = _load_raw_data_from_sql(asset_id)
    return frame[frame.day == date][colname].values[0]


def expected_bar_values_2d(dates, asset_info, colname):
    """
    Return an 2D array containing cls.expected_value(asset_id, date,
    colname) for each date/asset pair in the inputs.

    Values before/after an assets lifetime are filled with 0 for volume and
    NaN for price columns.
    """
    if colname == 'volume':
        dtype = uint32
        missing = 0
    else:
        dtype = float64
        missing = float('nan')

    assets = asset_info.index

    data = full((len(dates), len(assets)), missing, dtype=dtype)
    for j, asset in enumerate(assets):
        start = asset_start(asset_info, asset)
        end = asset_end(asset_info, asset)
        for i, date in enumerate(dates):
            # No value expected for dates outside the asset's start/end
            # date.
            if not (start <= date <= end):
                continue
            data[i, j] = expected_bar_value(asset, date, colname)
    return data

def load_stock_daily_raw_data(codes, field, start, end):
    """
    读取本地数据库股票日线单列原始数据
    
    >>> load_stock_daily_raw_data(['000001','000002'], 'close', '2017-11-27', '2017-11-30')
    array([[ 13.93,  31.52],
           [ 13.7 ,  30.75],
           [ 13.82,  33.82],
           [ 13.38,  31.22]])    
    """
    expr = STOCK_DB.stock_dailies
    cols = ['code','date'] + [field]
    dfs = [expr[expr.code == code][cols] for code in codes]
    prices = odo(df, pd.DataFrame)
    prices.set_index('date', inplace=True)
    raw = prices[start:end]
    return raw.pivot(columns='code').values


def select_by_sid(df, sid):
    """
    按`sid`选择pipeline输出结果


    假设pipeline输出
                                                        股利
    ---------------------------------------------------------
    2012-01-04 00:00:00+00:00 Equity(000001 [平安银行]) 0.10
                              Equity(000002 [万 科Ａ])  0.13
                              Equity(000004 [国农科技]) 0.03
                              Equity(000005 [世纪星源]) 0.00
                              Equity(000006 [深振业Ａ]) 0.04
    ......
    2017-02-10 ......

    筛选2006(002006)，下采样`result.resample('AS')`结果：

                              股利
    -------------------------------
    2012-01-01 00:00:00+00:00 0.10
    2013-01-01 00:00:00+00:00 0.00
    2014-01-01 00:00:00+00:00 0.00
    2015-01-01 00:00:00+00:00 0.00
    2016-01-01 00:00:00+00:00 0.00
    2017-01-01 00:00:00+00:00 0.02
    """
    res = df[df.index.map(lambda s:int(s[1])) == sid].copy()
    res.reset_index(level=1, drop=True, inplace=True)
    return res