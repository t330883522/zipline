"""
使用bcolz存储数据

设定定期计划任务，自动完成sql数据到bcolz的格式转换
使用bcolz默认参数压缩，初步测试结果表明，大型数据加载至少提速一倍

"""
import os
from shutil import rmtree
import pandas as pd
from functools import partial
from logbook import Logger
from odo import odo
import re

from cswd.utils import data_root

from .base import STOCK_DB
from .utils import _normalize_ad_ts_sid
from .categories import get_concept_categories

logger = Logger(__name__)

INCLUDE_TABLES = ['adjustments', 'margins', 'finance_reports', 'concept_stocks']

QUARTERLY_TABLES = ['finance_reports','shareholder_histoires']

LABEL_MAPS = {'date':'asof_date','code':'sid'}

concept_id_pat = re.compile('\d{6}')

def _bcolz_table_path(table_name):
    """bcolz文件路径"""
    root_dir = os.path.join(data_root('data'), 'bcolz')
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    path_ = os.path.join(root_dir, '{}.bcolz'.format(table_name))
    return path_


def normalized_dividend_data(df):
    """处理现金分红（调整到年初）"""
    raw_df = df[['code','date','amount']]
    raw_df.set_index('date',inplace=True)
    res = raw_df.groupby('code').resample('AS').sum().fillna(0.0)
    return res.reset_index()


def normalized_finance_reports_data(df):
    """财务报告及指标数据"""
    res = df.drop(['last_updated', 'announcement_date'], axis = 1)
    return res.fillna(value = 0.0)


def normalized_margins_data(df):
    """融资融券数据"""
    res = df.drop('last_updated', axis = 1)
    res = res.rename(columns = {'short_balance_amount':'total_balance'})
    return res

def _transfer_fun(sub_df, ids):
    """股票概念数据按股票代码分组后使用的转换函数"""
    out = sub_df.pivot(index='date', columns='concept_id', values='concept_id')
    out.fillna(method='bfill', inplace=True)
    out.replace(concept_id_pat, value=True, inplace=True)
    out.reindex(columns = ids, fill_value=False)
    return out

def normalized_stock_concept_data(df):
    """
    处理股票概念数据

    Notes
    -------
		f(sid, asof_date) -> bool
        腾讯概念分类含有诸如`当日涨停`等，无法适应ffill
        只能累积一段时期的数据后，才能使用。
    """
    df.drop('last_updated', axis = 1, inplace = True)
    ids = df.concept_id.unique()
    grouped = df.groupby('code')
    func = partial(_transfer_fun, ids = ids)
    out = grouped.apply(func).fillna(value=False)
    # bcolz列名称不得以数子开头，调整为'C' + 编码
    out.columns = ['C{}'.format(x) for x in out.columns]
    out = out.reset_index()
    return out

def _convert_factory(table_name):
    """转换函数工厂"""
    if table_name == 'adjustments':
        return normalized_dividend_data
    elif table_name == 'margins':
        return normalized_margins_data
    elif table_name == 'finance_reports':
        return normalized_finance_reports_data
    elif table_name == 'concept_stocks':
        return normalized_stock_concept_data
    raise NotImplementedError(table_name)


def _write_one_table(expr, ndays=0):
    """
    以bcolz格式写入表达式数据

    转换步骤：
        1. 读取表数据转换为pd.DataFrame

    """
    table_name = expr._name

    # 整理数据表

    # 1. 读取表
    df = odo(expr, pd.DataFrame)

    # 2. 调整转换数据
    processed = _convert_factory(table_name)(df)
           
    # 3. 列名称及类型标准化
    out = _normalize_ad_ts_sid(processed, ndays)

    # 转换为bcolz格式并存储
    rootdir = _bcolz_table_path(table_name)
    if os.path.exists(rootdir):
        rmtree(rootdir)
    odo(out, rootdir)
    logger.info('File was saved in {}'.format(rootdir))


def convert_sql_data_to_bcolz():
    """
    将部分sql数据表以bcolz格式存储，提高数据集加载速度

    Notes:
    ------
        写入财务报告数据时，默认财务报告公告日期为报告期后45天
    """
    for table in STOCK_DB.fields:
        if table in INCLUDE_TABLES:
            logger.info('preprocessing table:"{}"'.format(table))
            expr = STOCK_DB[table]
            ndays = 45 if table in QUARTERLY_TABLES else 0
            _write_one_table(expr, ndays)
