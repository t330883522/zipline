import pandas as pd
import numpy as np
from functools import partial

from zipline.pipeline.factors import CustomFactor
from zipline.pipeline import Fundamentals
from zipline.pipeline.data import USEquityPricing
from zipline.utils.input_validation import expect_types
from zipline.utils.numpy_utils import changed_locations

np.seterr(divide='ignore', invalid='ignore')

def IsStock():
    """股票过滤器"""
    return USEquityPricing.tmv.latest > 0.

#------------------------------------------------------------------------------
#   股票基础信息
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
    inputs = [Fundamentals.finance_reports.A095,
              USEquityPricing.cmv,
              USEquityPricing.tmv]
    window_length = 1
    mask = IsStock()
    def compute(self, today, assets, out, st, cmv, tmv):
        out[:] = tmv / cmv * st * 10000

#------------------------------------------------------------------------------
#   统计指标
#------------------------------------------------------------------------------

class HistoricalZscore(CustomFactor):
    """
    纵向zscore值因子。反应当前数据在窗口数据集中相对位置。
    
    Notes:
    ------
        区别于`zipline`计算，只与列自身历史数据相关。
    """
    #mask = IsStock()
    #mask = StaticSids([1, 2, 333])
    #outputs = ['close_zscore', 'volume_zscore']
    #inputs = [USEquityPricing.close, USEquityPricing.volume]
    window_safe = True
    def _validate(self):
        super(HistoricalZscore, self)._validate()
        if self.window_length < 3:
            raise ValueError(
                "`HistoricalZscore`窗口长度至少为3，少于3没有实际意义。"
                "实际输入窗口数量 {window_length}"
                "".format(window_length=self.window_length)
            )  
        if len(self.inputs) != 1:
            raise ValueError(
                "`HistoricalZscore`只接受单列，但输入了{length}列。"
                "".format(length=len(self.inputs))
            ) 
            
    def compute(self, today, assets, out, values):
        out[:] = (values[-1] - np.nanmean(values,0)) / np.nanstd(values,0)


#------------------------------------------------------------------------------
#   股票技术指标
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
#   股票财务指标
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
    inputs = [Fundamentals.finance_reports.B001, 
              Fundamentals.finance_reports.asof_date]
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
              Fundamentals.finance_reports.B001, 
              Fundamentals.finance_reports.asof_date]

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
              Fundamentals.finance_reports.B001, 
              Fundamentals.finance_reports.asof_date]

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
              Fundamentals.finance_reports.B044, 
              Fundamentals.finance_reports.asof_date]

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
              Fundamentals.finance_reports.B045, 
              Fundamentals.finance_reports.asof_date]

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
              Fundamentals.finance_reports.D002, Fundamentals.finance_reports.asof_date]
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


# # 常用别名
PS = PriceToSales
PE = PriceToEarnings
DPE = PriceToDilutedEarnings
PB = PriceToBook
PTTMS = PriceToTTMSales
DividendYield = PriceToDividendYield

#------------------------------------------------------------------------------
#   成长性
#------------------------------------------------------------------------------

class SalesGrowth(CustomFactor):
    """
    主营业务收入增长率

    Notes:
    High value represents large growth (short term)
    """
    inputs = [Fundamentals.finance_reports.G001]
    window_length = 1

    def compute(self, today, assets, out, growth):
        out[:] = growth

# 12-month Sales Growth
#class Sales_Growth_12M(CustomFactor):
#    """
#    12-month Sales Growth:

#    Increase in total sales over 12 months
#    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

#    Notes:
#    High value represents large growth (long term)
#    """
#    inputs = [morningstar.income_statement.total_revenue]
#    window_length = 252

#    def compute(self, today, assets, out, sales):
#        out[:] = ((sales[-1] * 4) - (sales[0]) * 4) / (sales[0] * 4)

## 12-month EPS Growth
#class EPS_Growth_12M(CustomFactor):
#    """
#    12-month Earnings Per Share Growth:

#    Increase in EPS over 12 months
#    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

#    Notes:
#    High value represents large growth (long term)
#    """
#    inputs = [morningstar.earnings_report.basic_eps]
#    window_length = 252

#    def compute(self, today, assets, out, eps):
#        out[:] = (eps[-1] - eps[0]) / eps[0]


#------------------------------------------------------------------------------
#   股票指数相关
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