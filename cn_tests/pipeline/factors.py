import unittest

from numpy.testing import assert_array_almost_equal
import numpy as np

from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.filters import StaticSids

from zipline.pipeline.builtin import HistoricalZScore
from zipline.research import run_pipeline


def make_pipeline():
    return Pipeline(
        columns = {
            'close_zscore':HistoricalZScore(window_length=4, inputs=[USEquityPricing.close]),
            'volume_zscore':HistoricalZScore(window_length=4, inputs=[USEquityPricing.volume]),
        },
        screen = StaticSids([1,2])
    )

class Test_factors(unittest.TestCase):
    def test_HistoricalZscore(self):
        # 因读取原始数据时，并没有考虑调整项。故不得选择送转股期间测试。
        # 如果原始数据做同样的调整处理，才会通过测试。
        # 关注点：按列计算z值， 单列输入
        start = '2017-11-20'
        end = '2017-12-1'
        stock_codes = ['000001','000002']
        df = run_pipeline(make_pipeline(), start, end)

        result = df.loc[end].values

        expected = np.array([[-1.59106704, -1.23624812],
                             [-0.51367549,  1.18221339]])

        assert_array_almost_equal(result, expected)

        with self.assertRaises(ValueError):
            HistoricalZScore(window_length=2, inputs=[USEquityPricing.close])

        with self.assertRaises(ValueError):
            HistoricalZScore(window_length=4, inputs=[USEquityPricing.close,
                                                      USEquityPricing.volume])


if __name__ == '__main__':
    unittest.main()
