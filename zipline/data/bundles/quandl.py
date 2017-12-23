"""
构造完整股票日线数据集
"""

import pandas as pd

from logbook import Logger

from cswd.sqldata.core import (fetch_single_stock_equity, 
                               fetch_single_index_equity,
                               fetch_single_quity_adjustments, 
                               fetch_symbol_metadata_frame)

from . import core as bundles
from ..constants import ADJUST_FACTOR, TEST_SIDS

log = Logger(__name__)

# # 增加处理函数
def _adjusted_raw_data(raw_df):
    """调整原始数据单位，转换为unit32类型"""
    data = raw_df.copy()
    # 增强兼容性
    for col in data.columns:
        if col in ADJUST_FACTOR.keys():
            data.loc[:, col] = data[col] * ADJUST_FACTOR.get(col,1)
    return data


def _update_splits(splits, asset_id, origin_data):
    # 分红派息共用一个数据对象，防止相互影响
    raw_data = origin_data.copy()
    split_ratios = raw_data.ratio
    # 调整适应于zipline算法
    df = pd.DataFrame({'ratio': 1 / (1 + split_ratios[split_ratios != 0])})
    if df.empty:
        return
    df.index.name = 'effective_date'
    df.reset_index(inplace=True)
    df['sid'] = asset_id
    splits.append(df)


def _update_dividends(dividends, asset_id, origin_data):
    # 分红派息共用一个数据对象，防止相互影响
    raw_data = origin_data.copy()
    divs = raw_data.amount
    df = pd.DataFrame({'amount': divs[divs != 0]})
    if df.empty:
        return
    df.index.name = 'ex_date'
    # 确保index一致，且应放置在reset_index之前完成
    df['record_date'] = raw_data.loc[df.index, 'record_date']
    df['pay_date'] = raw_data.loc[df.index, 'pay_date']
    df.reset_index(inplace=True)
    df['sid'] = asset_id
    # 原始数据中存在 'record_date' 'pay_date'列，但没有'declared_date'列
    df['declared_date'] = pd.NaT
    dividends.append(df)


def gen_symbol_data(symbol_map,
                    sessions,
                    splits,
                    dividends):
    for asset_id, symbol in symbol_map.iteritems():
        if len(symbol) == 7:
            # 指数
            raw_data = fetch_single_index_equity(
                symbol,
                start_date=sessions[0],
                end_date=sessions[-1],
            )
        else:
            # 股票
            raw_data = fetch_single_stock_equity(
                symbol,
                start_date=sessions[0],
                end_date=sessions[-1],
            )

            # 获取原始调整数据
            raw_adjustment = fetch_single_quity_adjustments(symbol,
                                                            start_date=sessions[0],
                                                            end_date=sessions[-1]
                                                            )
            # 调整index的类型
            raw_adjustment.index = pd.to_datetime(raw_adjustment.index)

            _update_splits(splits, asset_id, raw_adjustment)
            _update_dividends(dividends, asset_id, raw_adjustment)

        # 不得包含涨跌幅列（int32）
        raw_data.drop('change_pct',axis = 1, inplace = True)
        # 调整数据精度
        raw_data = _adjusted_raw_data(raw_data)
        # 时区调整
        raw_data = raw_data.reindex(
            sessions.tz_localize(None),
            copy=False,
        ).fillna(0.0)
        yield asset_id, raw_data

@bundles.register('cnstock', calendar_name = 'SZSH', minutes_per_day = 330)
def quandl_bundle(environ,
                  asset_db_writer,
                  minute_bar_writer,
                  daily_bar_writer,
                  adjustment_writer,
                  calendar,
                  start_session,
                  end_session,
                  cache,
                  show_progress,
                  output_dir):
    """Build a zipline data bundle from the cnstock dataset.
    """
    metadata = fetch_symbol_metadata_frame()

    symbol_map = metadata.symbol
    sessions = calendar.sessions_in_range(start_session, end_session)

    log.info('日线数据集（股票及指数数量：{}）'.format(len(symbol_map)))

    # 写入股票元数据
    if show_progress:
        log.info('Generating asset metadata.')
    asset_db_writer.write(metadata)

    splits = []
    dividends = []
    daily_bar_writer.write(
        gen_symbol_data(
            symbol_map,
            sessions,
            splits,
            dividends,
        ),
        show_progress=show_progress,
    )

    adjustment_writer.write(
        splits=pd.concat(splits, ignore_index=True),
        dividends=pd.concat(dividends, ignore_index=True),
    )


@bundles.register('.test', calendar_name = 'SZSH', minutes_per_day = 330)
def test_bundle(environ,
                asset_db_writer,
                minute_bar_writer,
                daily_bar_writer,
                adjustment_writer,
                calendar,
                start_session,
                end_session,
                cache,
                show_progress,
                output_dir):
    """Build a zipline test data bundle from the cnstock dataset.
    """
    # 测试集
    metadata = fetch_symbol_metadata_frame().loc[TEST_SIDS,:]
    sessions = calendar.sessions_in_range(start_session, end_session)

    symbol_map = metadata.symbol
    log.info('生成测试数据集（共{}只）'.format(len(symbol_map)))
    splits = []
    dividends = []

    asset_db_writer.write(metadata)

    daily_bar_writer.write(
        gen_symbol_data(
            symbol_map,
            sessions,
            splits,
            dividends,
        ),
        show_progress=show_progress,
    )

    adjustment_writer.write(
        splits=pd.concat(splits, ignore_index=True),
        dividends=pd.concat(dividends, ignore_index=True),
    )