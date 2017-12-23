"""
与估值相关的因子

注：
---
    区别于严格意义上的估值

基本概念及公式
    NOPLAT（Net Operating Profits Less Adjusted Taxes）
        扣除与核心经营活动有关的所得税后公司核心经营活动产生的利润
    IC（Invested Capital）
        公司在核心经营活动上已投资的累计数额
    净投资
        净投资 = 投入资本<t+1> - 投入资本<t>
    FCF（Free Cash Flow）
        扣除新增资本投资后公司核心经营活动产生的现金流
        FCF = NOPLAT - 净投资
    ROIC
        ROIC = NOPLAT / IC
    IR（Investment Rate）
        IR = 净投资 / NOPLAT
    WACC（Weighted Average Cost of Capital）
        加权平均资本成本
    g（增长率）
        指NOPLAT和现金流的增长率

    价值 = NOPLAT<t=1> * (1 - (g / ROIC)) / (WACC - g)

投入资本
    -- 投入资本包含了自有投入及借款投入

    经营资产的主要构成：
        应收账款
        存货和不动产
        厂房和设备（PP&E）
    经营负债的主要构成：
        应付账款
        应付工资
        有息负债（D）：如应付票据和长期借款
    权益（E）主要构成：
        普通股、优先股、留存收益
        

    经营资产 = 经营债务 + 负债 + 权益
    经营资产 - 经营债务 = 投入资本 = 负债 + 权益
    经营资产（OA）+ 非经营资产（NOA） = 经营债务（OL） + 负债和负债等价物（D+DE） + 权益和权益等价物（E+EE）
    投入资本（OA - OL）+ 非经营资产（NOA） = 总投入资金 = 负债和负债等价物（D+DE） + 权益和权益等价物（E+EE）

NOPLAT简表
__________________________
收入                  1000
经营成本             （700）
折旧                 （20）
                    -------
经营利润              280
经营税               （40）   假定所得税25%
                    --------
NOPLAT               210

FCF（自由现金流）
    FCF = NOPLAT + 非现金经营费用 - 投入资本的增量部分
    计算FCF是以NOPLAT为起点，而现金流量表则以净收入为起点

FCF简表
__________________________
NOPLAT                210
折旧                    20    非现金流费用
                    -------
总现金流               230

存货的减少（增加）   （25）
应付账款的增加（减少）  25
资本支出             （70）   
                    --------
总投资               （70）

自由现金流             160
"""

from zipline.pipeline import Fundamentals
from zipline.pipeline.factors import CustomFactor

#------------------------------------------------------------------------------
#                                  非金行业                                   #
#------------------------------------------------------------------------------

class EBIT(CustomFactor):
    """
    息税前利润（扣除利息、税收前利润）
    
    公式
    -----
        营业利润 + 财务费用

    """
    window_length = 1
    inputs = [Fundamentals.income_statement.B033,
              Fundamentals.income_statement.B023,
             ]
    def compute(self, today, assets, out, B033, B023):
        out[:] = B033 + B023


class EBITA(CustomFactor):
    """
    息税摊销前利润（扣除利息、摊销、税收前的利润）
    Earnings Before Interest, Taxes, Depreciation and Amortization
    公式
    -----
        营业利润 + 财务费用 + 无形资产摊销 + 长期待摊费用摊销

    """
    window_length = 1
    inputs = [Fundamentals.income_statement.B033,
              Fundamentals.income_statement.B023,
              Fundamentals.cash_flow.C062,
              Fundamentals.cash_flow.C063,
             ]
    def compute(self, today, assets, out, B033, B023, C062, C063):
        out[:] = B033 + B023 + C062 + C063


class NOPLAT(CustomFactor):
    """
    扣除与核心经营活动有关的所得税后公司核心经营活动产生的利润
    
    公式
    -----
        EBIT - 经营现金税

    """
    window_length = 1
    inputs = [Fundamentals.income_statement.B033,
              Fundamentals.income_statement.B023,
              Fundamentals.income_statement.B037,
              Fundamentals.income_statement.B038,
             ]
    def compute(self, today, assets, out, B033, B023, B037, B038):
        t = B038 / B037
        out[:] = (B033 + B023) * (1 - t)


class FCF(CustomFactor):
    """
    自由现金流
    
    公式
    -----
        NOPLAT + 非现金经营费用 - 投入资本的增量部分

    """
    window_length = 1
    inputs = [Fundamentals.income_statement.B033,
              Fundamentals.income_statement.B023,
              Fundamentals.income_statement.B037,
              Fundamentals.income_statement.B038,
              Fundamentals.cash_flow.C061,
              Fundamentals.cash_flow.C040,
             ]
    def compute(self, today, assets, out, B033, B023, B037, B038,
                C061, C040):
        t = B038 / B037
        noplat = (B033 + B023) * (1 - t)
        out[:] = noplat + C061 - C040