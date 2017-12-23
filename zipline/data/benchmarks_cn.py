"""
读取本地指数数据作为基准收益率
"""
import pandas as pd
from cswd.sqldata.stock_index_daily import StockIndexDailyData

def get_benchmark_returns(symbol, first_date, last_date):
    """
    Get a Series of benchmark returns from local db with `symbol`.
    Default is `000300`.

    Parameters
    ----------
    symbol : str
        Benchmark symbol for which we're getting the returns.
    first_date : datetime_like
        First date for which we want to get data.
    last_date : datetime_like
        Last date for which we want to get data.

    first_date is **not** included because we need the close from day N - 1 to
    compute the returns for day N.
    """
    # 数据库日期类型为datetime.date
    first_date, last_date = first_date.date(), last_date.date()
    data = StockIndexDailyData.query_by_code(symbol).loc[first_date:last_date,'close']
    df = data.sort_index().pct_change(1).iloc[1:]
    df.index = pd.to_datetime(df.index, utc=True)
    return df