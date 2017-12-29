"""
简单版本（仅仅处理日线）
"""
import pandas as pd
import datetime as dt

from pytz import timezone

from cswd.constants import MARKET_START
from cswd.dataproxy.data_proxies import non_trading_days_reader

from zipline.utils.calendars import TradingCalendar


class SZSHExchangeCalendar(TradingCalendar):
    """
    TradingCalendar子类
    
    注：
        1、不考虑午间休息
        2、自然日历 - 交易日历 = 非交易日
    """

    @property
    def name(self):
        return "SZSH"

    @property
    def tz(self):
        return timezone('Asia/Shanghai')

    @property
    def open_time(self):
        return dt.time(9, 31)

    @property
    def close_time(self):
        return dt.time(15)

    @property
    def adhoc_holidays(self):
        start_default = MARKET_START.tz_localize('UTC') - pd.Timedelta(days = 1)
        start_default = start_default.normalize()
        end_default = pd.Timestamp('today', tz='UTC') + pd.Timedelta(days=365)
        end_default = end_default.normalize()
        return non_trading_days_reader.read(start_default, end_default)

    # 日线级别用不上
    #@property
    #def special_closes_adhoc(self):
    #    """
    #    提前收盘时间（历史熔断）

    #    Returns
    #    -------
    #    list: List of (time, DatetimeIndex) tuples that represent special
    #     closes that cannot be codified into rules.
    #    """
    #    return [(dt.time(13,33), ['2016-01-04']),
    #            (dt.time(9,59), ['2016-01-07']),]

    @property
    def actual_last_session(self):
        """最后交易日"""
        now = pd.Timestamp.utcnow()
        trading_days = self.all_sessions
        last_trading_session = trading_days[trading_days.get_loc(now, method='ffill')]
        return last_trading_session
