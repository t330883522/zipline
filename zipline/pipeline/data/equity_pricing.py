"""
Dataset representing OHLCV data.

增加非调整列
该部分不参与因分红派息的调整，即：
即使股票在某日实施1：1送股，也不会影响到这些列的值

"""
from zipline.utils.numpy_utils import float64_dtype

from .dataset import Column, DataSet


class USEquityPricing(DataSet):
    """
    Dataset representing daily trading prices and volumes.
    """
    open = Column(float64_dtype)
    high = Column(float64_dtype)
    low = Column(float64_dtype)
    close = Column(float64_dtype)
    volume = Column(float64_dtype)
    prev_close = Column(float64_dtype)
    turnover = Column(float64_dtype)
    amount = Column(float64_dtype)
    tmv = Column(float64_dtype)
    cmv = Column(float64_dtype)

# 别名
CNEquityPricing = USEquityPricing
