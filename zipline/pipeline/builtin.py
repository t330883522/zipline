"""
在常用数据集基础上的自定义因子、过滤、分类
"""

import numpy as np
import pandas as pd
import talib as ta

from zipline.utils.numpy_utils import changed_locations
from zipline.utils.input_validation import expect_types

from .fundamentals.core import Fundamentals, _SECTOR_NAMES
from .data.equity_pricing import USEquityPricing

from .factors import SimpleMovingAverage
from .factors.factor import CustomFactor

np.seterr(divide='ignore', invalid='ignore')

#------------------------------------------------------------------------------
#                                  总体过滤                                   #
#------------------------------------------------------------------------------

def IsStock():
    """股票过滤器"""
    return USEquityPricing.tmv.latest > 0.


def _common_filter(exclude_st = True):
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
    --------
    zipline.pipeline.Filter
        Filter for the set_screen of a Pipeline to create an
        ideal trading universe

    过滤条件
    ---------
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


def make_us_equity_universe(target_size, 
                            rankby, 
                            groupby, 
                            max_group_weight = 0.3, 
                            mask = _common_filter(), 
                            smoothing_fun = lambda f: f.downsample('month_start')):
    # 按组数量是错误的，待修正
    group_size = int(target_size * max_group_weight)
    # 生成一个新的因子
    ranked = rankby.rank(mask = mask, groupby = groupby)
    group_top = ranked.top(group_size, groupby = groupby)
    smoothed = smoothing_fun(group_top)
    return smoothed


def default_us_equity_universe_mask(minimum_market_cap=500000000):
    """默认市值超过最低限的总体过滤器"""
    primary_share = IsStock()
    cap_ok = USEquityPricing.tmv.latest > minimum_market_cap
    return primary_share & cap_ok


def Q500US(minimum_market_cap=500000000, exclude_st=True):
    """
    一个默认的总体，每天大约有500个A股股票
    在每个日历月初选择成分股，选择500个“可交易”股票，
    以200天的平均成交量计算，任何一个行业的股票上限为
    总体的30％。

    returns
    --------
    zipline.pipeline.Filter
        Filter for the set_screen of a Pipeline to create an
        ideal trading universe

    过滤条件
    ---------
        1. 存在市值（未退市，且不为指数），且不少于`minimum_market_cap`
        2. 正常交易的股票（不得包含`退`（退市）)
        3. 如过滤ST，则所有股票简称含有ST字样被过滤掉
        4. 前一交易日未停牌（以成交量判断）
    """
    tradable_filter = _common_filter(exclude_st)
    # 以总市值来判定
    market_cap = USEquityPricing.tmv.latest.downsample('month_start')
    cap_ok = market_cap >= minimum_market_cap
    trading_ok = USEquityPricing.volume.latest > 0.
    # 500 * 0.3 = 150
    # 还需要分组，控制每组数量，取前500
    return tradable_filter & cap_ok & trading_ok


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

#------------------------------------------------------------------------------
#                                  上市日期                                   #
#------------------------------------------------------------------------------

class NDays(CustomFactor):
    """上市天数因子"""

    # 输入shape = (m days, n assets)
    inputs = [Fundamentals.time_to_market]
    window_length = 1
    mask = IsStock()

    def compute(self, today, assets, out, ts):
        baseline = today.tz_localize(None)
        days = [(baseline - pd.Timestamp(x)).days for x in ts[0]]
        out[:] = days

#------------------------------------------------------------------------------
#                                  季度股本                                   #
#------------------------------------------------------------------------------

class SharesTotal(CustomFactor):
    """
    总股本
    
    Notes
    ------
        不同于日常交易的总股本，季度时点数据。
        日常交易的总股本：
        context.current(sids, 'tmv') / context.current(sids,'close'])
    """

    # 输入shape = (m days, n assets)
    inputs = [Fundamentals.balance_sheet.A095]
    window_length = 1
    mask = IsStock()
    # 原始数据单位：万元
    def compute(self, today, assets, out, st):
        out[:] = st * 10000


class SharesOutstanding(CustomFactor):
    """流通股本"""
    inputs = [Fundamentals.balance_sheet.A095,
              USEquityPricing.cmv,
              USEquityPricing.tmv]
    window_length = 1
    mask = IsStock()
    def compute(self, today, assets, out, st, cmv, tmv):
        out[:] = tmv / cmv * st * 10000


class MarketCap(CustomFactor):
    """流通市值"""
    inputs = [USEquityPricing.cmv]
    # 如果希望使用总市值
    #inputs = [USEquityPricing.tmv]
    def compute(self, today, assets, out, data):
        out[:] = data[-1]

#------------------------------------------------------------------------------
#                                  行    业                                   #
#------------------------------------------------------------------------------
class Sector(CustomFactor):
    """部门行业"""
    window_length = 1
    inputs = [Fundamentals.cninfo.sector]
    SECTOR_NAMES = _SECTOR_NAMES
    def compute(self, today, assets, out, data):
        out[:] = data

#------------------------------------------------------------------------------
#                                  财务指标                                   #
#------------------------------------------------------------------------------

def quarterly_multiplier(dates):
    """
    季度乘数，用于预测全年数据（季度数据辅助函数）

    参数
    ------
        dates：季度数据报告日期（区别于公告日期）

    Notes:
    ------
        假设前提：财务数据均匀分布

    例如：
        第三季度累计销售收入为6亿，则全年预测为8亿。

        乘数 = 4 / 3
    """
    func = lambda x:x.quarter
    if dates.ndim == 1:
        qs = pd.Series(dates).map(func)
    else:
        qs = pd.DataFrame(dates).applymap(func)
    return 4 / qs


class TTMSales(CustomFactor):
    """
    最近一年营业总收入

    Notes:
    ------
    trailing 12-month (TTM)
    """
    inputs = [Fundamentals.income_statement.B001, 
              Fundamentals.income_statement.asof_date]
    window_length = 252
    mask = IsStock()

    def compute(self, today, assets, out, sales, asof_date):
        # 原始数据已经reindex全部季度，所以只需要单个资产的asof_date
        idx = changed_locations(asof_date.T[0], include_first=True)[1:]
        adj = quarterly_multiplier(asof_date[idx])
        # 将各季度调整为年销售额，然后再平均
        res = np.multiply(sales[idx], adj).mean()
        # 收入单位为万元
        out[:] = res * 10000


class PriceToTTMSales(CustomFactor):
    """
    股价销售比率 = 股价 / 每股TTM销售

    Notes:
    ------
    trailing 12-month (TTM)
    股价销售比率 = 股价 / 每股TTM销售
                 = 股价 * 总股本 / （每股TTM销售 * 总股本)
                 = 总市值 / 最近一年总销售额
    """
    inputs = [USEquityPricing.tmv,
              Fundamentals.income_statement.B001, 
              Fundamentals.income_statement.asof_date]

    window_length = 252
    mask = IsStock()

    def compute(self, today, assets, out, tmv, sales, asof_date):
        # 原始数据已经reindex全部季度，所以只需要单个资产的asof_date
        # 年度窗口长度一般取252，包含了五个季度。只需要最近四季度数据
        idx = changed_locations(asof_date.T[0], include_first=True)[1:]
        adj = quarterly_multiplier(asof_date[idx])
        # 将各季度调整为年销售额，然后再平均
        res = np.multiply(sales[idx], adj).mean()
        # 收入单位为万元
        out[:] = tmv[-1] / res * 10000


class PriceToSales(CustomFactor):
    """
    PS（市销率）
    Price to Sales Ratio:

    Notes:
    Low P/S Ratio suggests that an equity cheap
    Differs substantially between sectors
    """
    inputs = [USEquityPricing.tmv, 
              Fundamentals.income_statement.B001, 
              Fundamentals.income_statement.asof_date]

    window_length = 1
    mask = IsStock()

    def compute(self, today, assets, out, tmv, sales, asof_date):
        # 收入单位为万元
        adj = quarterly_multiplier(asof_date[-1])
        out[:] = tmv[-1] / (sales[-1] * adj * 10000.)


class PriceToEarnings(CustomFactor):
    """
    PE（市盈率）= 股价 / 每股基本收益

    """
    inputs = [USEquityPricing.close, 
              Fundamentals.income_statement.B044, 
              Fundamentals.income_statement.asof_date]

    window_length = 1
    mask = IsStock()

    def compute(self, today, assets, out, close, eps, asof_date):
        adj = quarterly_multiplier(asof_date[-1])
        eps[eps == 0.] = np.nan
        out[:] = close[-1] / (eps[-1] * adj)


class PriceToDilutedEarnings(CustomFactor):
    """
    股价稀释每股收益比率 = 股价 / 每股稀释收益

    Notes:
    ------

    Low P/Diluted Earnings Ratio suggests that equity is cheap
    Differs substantially between sectors
    """
    inputs = [USEquityPricing.close, 
              Fundamentals.income_statement.B045, 
              Fundamentals.income_statement.asof_date]

    window_length = 1
    mask = IsStock()

    def compute(self, today, assets, out, close, deps, asof_date):
        adj = quarterly_multiplier(asof_date[-1])
        deps[deps == 0.] = np.nan
        out[:] = close[-1] / (deps[-1] * adj)


class PriceToBook(CustomFactor):
    """
    调整后市净率（PB）= 股价 / 每股净资产 * 季度调整因子

    Notes:
    ------
    Low P/B Ratio suggests that equity is cheap
    Differs substantially between sectors
    """
    inputs = [USEquityPricing.close, 
              Fundamentals.key_financial_indicators.D002, 
              Fundamentals.key_financial_indicators.asof_date]
    window_length = 1
    mask = IsStock()

    def compute(self, today, assets, out, close, bs, asof_date):
        adj = quarterly_multiplier(asof_date[-1])
        out[:] = close[-1] / (bs[-1] * adj)


class PriceToDividendYield(CustomFactor):
    """
    股息率 = 每股股利（年汇总） / 每股股价

    Notes:
    ------
    每股股利为年度股利之和，只要当年存在分配股利，则分配日视同为年初。
    """
    inputs = [USEquityPricing.close, Fundamentals.dividend]
    window_length = 1
    mask = IsStock()

    def compute(self, today, assets, out, close, dividend):
        dividend[dividend == 0.] = np.nan
        out[:] = dividend / close

#------------------------------------------------------------------------------
#                                  技术相关                                   #
#------------------------------------------------------------------------------

@expect_types(df = pd.DataFrame)
def continuous_num(df):
    """计算连续数量"""
    msg = "数据框所有列类型都必须为`np.dtype('bool')`"
    assert all(df.dtypes == np.dtype('bool')), msg
    def compute(x):
        num = len(x)
        res = 0
        for i in range(num-1, 0, -1):
            if x[i] == 0:
                break
            else:
                res += 1
        return res
    return df.apply(compute, axis = 0)


class SuccessiveYZ(CustomFactor):
    """
    连续一字板数量

    returns
    -------
        1. 连续一字涨停数量
        2. 连续一字跌停数量

    Notes:
    ------
        截至当前连续一字板数量。
        1. 最高 == 最低
        2. 涨停：涨跌幅 >= 0.095
        3. 跌停：涨跌幅 <= -0.095
    """

    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    mask = IsStock()
    outputs = ['zt', 'dt']

    def compute(self, today, assets, out, high, low, close):
        is_zt = pd.DataFrame(close).pct_change() >= 0.095
        is_dt = pd.DataFrame(close).pct_change() <= -0.095
        is_yz = pd.DataFrame((high == low))
        yz_zt = is_zt & is_yz
        yz_dt = is_dt & is_yz
        out.zt = continuous_num(yz_zt).values
        out.dt = continuous_num(yz_dt).values

#------------------------------------------------------------------------------
#                                  指数相关                                   #
#------------------------------------------------------------------------------

class IndexBeta(CustomFactor):
    """
    股票相对沪深300指数的Beta系数

    Slope coefficient of 1-year regression of price returns against index returns
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

    Notes:
    High value suggests high market risk
    Slope calculated using regression MLE
    """
    inputs = [USEquityPricing.close]
    window_length = 252

    def compute(self, today, assets, out, close):

        # get index and calculate returns. 默认为沪深300
        benchmark_index = np.where((assets == 100300) == True)[0][0]
        benchmark_close = close[:, benchmark_index]
        benchmark_returns = (
            (benchmark_close - np.roll(benchmark_close, 1)) / np.roll(benchmark_close, 1))[1:]

        betas = []

        # get beta for individual securities using MLE
        for col in close.T:
            col_returns = ((col - np.roll(col, 1)) / np.roll(col, 1))[1:]
            col_cov = np.cov(col_returns, benchmark_returns)
            betas.append(col_cov[0, 1] / col_cov[1, 1])
        out[:] = betas

# Downside Beta
class Downside_Beta(CustomFactor):
    """
    Downside Beta:

    Slope coefficient of 1-year regression of price returns against negative index returns
    http://www.ruf.rice.edu/~yxing/downside.pdf

    Notes:
    High value suggests high exposure to the downmarket
    Slope calculated using regression MLE
    """
    inputs = [USEquityPricing.close]
    window_length = 252

    def compute(self, today, assets, out, close):

        # get index and calculate returns. 默认为沪深300
        benchmark_index = np.where((assets == 100300) == True)[0][0]
        benchmark_close = close[:, benchmark_index]
        benchmark_returns = (
            (benchmark_close - np.roll(benchmark_close, 1)) / np.roll(benchmark_close, 1))[1:]

        # days where benchmark is negative
        negative_days = np.argwhere(benchmark_returns < 0).flatten()

        # negative days for benchmark
        bmark_neg_day_returns = [benchmark_returns[i] for i in negative_days]

        betas = []

        # get beta for individual securities using MLE
        for col in close.T:
            col_returns = ((col - np.roll(col, 1)) / np.roll(col, 1))[1:]
            col_neg_day_returns = [col_returns[i] for i in negative_days]
            col_cov = np.cov(col_neg_day_returns, bmark_neg_day_returns)
            betas.append(col_cov[0, 1] / col_cov[1, 1])
        out[:] = betas


#------------------------------------------------------------------------------
#                                  模式识别                                   #
#------------------------------------------------------------------------------
class CDL_Patterns(CustomFactor):
    """
    模式识别（所有模式，多输出）

    .. code-block:: python
        cdls = CDL_Patterns()
        a = cdls.CDL2CROWS
        b = cdls.CDL3BLACKCROWS

    returns
    -------
        talib所有模式值Integer

    """
    window_length = 5 # 模式识别窗口最少为1，最多5。取5?
    inputs = [USEquityPricing.open, USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
    mask = IsStock()
    outputs = ta.get_function_groups()['Pattern Recognition']
    window_safe = True

    def _validate(self):
        super(CDL_Patterns, self)._validate()
        if self.window_length < 5:
            raise ValueError(
                "For pattern recognition,'CDL_Patterns' expected a window length of at least 5, but was "
                "given {window_length}. ".format(window_length=self.window_length)
            )

    def _one_pattern(self, out, func_name, opens, highs, lows, closes, N):
        """计算指定模式名称因子"""
        p_func = ta.abstract.Function(func_name)
        res = np.zeros(N)
        for i in range(N):
            inputs = {'open':opens[:,i],
                      'high':highs[:,i],
                      'low':lows[:,i],
                      'close':closes[:,i]}
            p_func.input_arrays = inputs
            res[i] = p_func.outputs[-1]
        setattr(out, func_name, res)

    def compute(self, today, assets, out, opens, highs, lows, closes):
        N = len(assets)
        for func_name in self.outputs:
            self._one_pattern(out, func_name, opens, highs, lows, closes, N)


# # 常用别名
PS = PriceToSales
PE = PriceToEarnings
DPE = PriceToDilutedEarnings
PB = PriceToBook
PTTMS = PriceToTTMSales
DividendYield = PriceToDividendYield
# 别名
Q500CN = Q500US