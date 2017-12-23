import unittest
from pandas.testing import assert_frame_equal

import pandas as pd

from zipline.component.research import run_pipeline
from zipline.pipeline import Fundamentals
from zipline.pipeline import Pipeline
from zipline.pipeline.filters import StaticSids

from cn_tests.utils import select_by_sid

def make_dividend_pipeline():
    dividend = Fundamentals.dividend.latest
    return Pipeline(
        columns = {
            '股利':dividend
        },
        screen = StaticSids([2006, 333])
    ) 

def make_finance_report_pipeline():
    return Pipeline(
        columns = {
            '货币资金(万元)':Fundamentals.finance_reports.A001.latest,
            '营业总收入(万元)':Fundamentals.finance_reports.B001.latest,
        },
        screen = StaticSids([2006, 333])
    ) 

def make_margin_pipeline():
    return Pipeline(
        columns = {
            '融资余额(万元)':Fundamentals.margin.long_balance_amount.latest,
            '融资融券余额(万元)':Fundamentals.margin.total_balance.latest,
        },
        screen = StaticSids([2006, 333])
    ) 

def make_concept_pipeline():
    
    C021196 = Fundamentals.concept.C021196.latest
    C021213 = Fundamentals.concept.C021213.latest
    C021095 = Fundamentals.concept.C021095.latest
    
    return Pipeline(
        columns = {
            '特种药':C021196,
            '大数据':C021213,
            '融资融券':C021095,
        },
        screen = StaticSids([977, 153])
    ) 

class Test_blaze(unittest.TestCase):

    def test_dividend_002006(self):
        # 重点测试，无现金分红年度，其值为0.0
        pipe = make_dividend_pipeline()
        df = run_pipeline(pipe, '2012-1-1','2017-1-10')
        result = select_by_sid(df, 2006).resample('AS').first()
        expected = pd.DataFrame({'股利':[0.10, 0.00, 0.00, 0.00,0.00,0.02]},
                                index = pd.date_range('2012', '2017',tz='utc',freq='AS'))
        assert_frame_equal(result, expected)

    def test_dividend_000333(self):
        pipe = make_dividend_pipeline()
        df = run_pipeline(pipe, '2014-1-1','2017-1-10')
        result = select_by_sid(df, 333).resample('AS').first()
        expected = pd.DataFrame({'股利':[2.0, 1.0, 1.2, 1.0]},
                                index = pd.date_range('2014', '2017',tz='utc',freq='AS'))
        assert_frame_equal(result, expected)

    def test_finance_report_000333(self):
        # 重点测试日期偏移。默认报告期后45天
        pipe = make_finance_report_pipeline()
        df = run_pipeline(pipe, '2017-2-14','2017-2-16')
        result = select_by_sid(df, 333)

        a001 = [1944482.0, 1793125.0, 1793125.0]
        b001 = [11707795.0, 15984170.0, 15984170.0]
        expected = pd.DataFrame({'货币资金(万元)':a001, '营业总收入(万元)':b001},
                                index = pd.date_range('2017-02-14', '2017-02-16',tz='utc'))

        assert_frame_equal(result, expected)

    def test_margin_000333(self):
        # 当天查询上一交易日数据
        pipe = make_margin_pipeline()
        df = run_pipeline(pipe, '2017-11-29','2017-11-30')
        result = select_by_sid(df, 333)
        # 测试数据类型float64
        long_amount = [2976903630.0, 3021909993.0]
        total_amount = [2994651757.0, 3038726032.0]
        expected = pd.DataFrame({'融资余额(万元)':long_amount, 
                                 '融资融券余额(万元)':total_amount},
                                index = pd.date_range('2017-11-29', '2017-11-30',tz='utc'))

        assert_frame_equal(result, expected)

    def test_stock_concept(self):
        pipe = make_concept_pipeline()
        start = '2017-11-27'
        end = '2017-11-30'
        df = run_pipeline(pipe, start,end)
        # 浪潮信息
        result_1 = select_by_sid(df, 977)      
        expected_1 = pd.DataFrame({'大数据':[True] * 4,
                                   '特种药':[False] * 4,
                                   '融资融券':[True] * 4,},
                                  index = pd.date_range(start, end, tz='utc'))

        assert_frame_equal(result_1, expected_1)

        # 丰原药业
        result_2 = select_by_sid(df, 153)      
        expected_2 = pd.DataFrame({'大数据':[False] * 4, 
                                   '特种药':[True] * 4,
                                   '融资融券':[False] * 4,},
                                  index = pd.date_range(start, end, tz='utc'))

        assert_frame_equal(result_2, expected_2)


if __name__ == '__main__':
    unittest.main()
