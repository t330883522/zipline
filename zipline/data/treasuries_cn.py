"""
国库券资金成本
"""
import pandas as pd
from cswd.sqldata.treasury import TreasuryData

def earliest_possible_date():
    """
    The earliest date for which we can load data from this module.
    """
    return pd.Timestamp('2006-3-1', tz = 'UTC')


def get_treasury_data(start_date, end_date):
    """读取期间数据"""
    df = TreasuryData.read(start_date, end_date)
    df.index = pd.to_datetime(df.index, utc=True)
    return df