from __future__ import division

import numpy as np
import pandas as pd
import toolz.curried.operator as op

from zipline import get_calendar
from zipline.data.bundles import ingest, load, bundles
from cswd.sqldata.core import (fetch_single_equity,
                          fetch_single_quity_adjustments)

from zipline.lib.adjustment import Float64Multiply
from zipline.testing import (
    test_resource_path,
    tmp_dir,
    patch_read_csv,
)
from zipline.testing.fixtures import (
    ZiplineTestCase,
    WithResponses,
)
from zipline.testing.predicates import (
    assert_equal,
)
from zipline.utils.functional import apply

from zipline.data.bundles.quandl import TEST_SIDS
from zipline.utils.symbol_sid_convert import to_sid, to_symbol

TEST_BUNDLE_NAME = '.test'

def _raw_data(stock_codes, start, end, columns):
    """按quandl测试数据格式组合"""
    dfs = []
    for stock_code in stock_codes:
        df = fetch_single_equity(stock_code, start, end)[list(columns)]
        adjustments = fetch_single_quity_adjustments(stock_code, start, end)
        # 如需要测试'volume'列，需调整单位
        try:
            df['volume'] = df['volume'] / 100
        except:
            pass
        df['sid'] = int(stock_code)
        ex_dividend = adjustments.reindex(df.index)['ratio']
        df['ex_dividend'] = 1 / (ex_dividend.fillna(0) + 1)
        dfs.append(df)
    return pd.concat(dfs)

class QuandlBundleTestCase(WithResponses,
                           ZiplineTestCase):
    symbols = to_symbol(TEST_SIDS)
    start_date = pd.Timestamp('2014-01', tz='utc')
    end_date = pd.Timestamp('2015-01', tz='utc')
    bundle = bundles[TEST_BUNDLE_NAME]
    calendar = get_calendar(bundle.calendar_name)
    columns = 'open', 'high', 'low', 'close', 'volume'

    def _expected_data(self, asset_finder):
        sids = {
            symbol: asset_finder.lookup_symbol(
                symbol,
                None,
            ).sid
            for symbol in self.symbols
        }

        # Load raw data local db.
        data = _raw_data(self.symbols,
                         self.start_date,
                         self.end_date,
                         self.columns)

        all_ = data.set_index(
            'sid',
            append=True,
        ).unstack()

        # fancy list comprehension with statements
        @list
        @apply
        def pricing():
            for column in self.columns:
                vs = all_[column].values
                if column == 'volume':
                    vs = np.nan_to_num(vs)
                yield vs

        # the first index our written data will appear in the files on disk
        start_idx = (
            self.calendar.all_sessions.get_loc(self.start_date, 'ffill') + 1
        )

        ######修改到此处

        # convert an index into the raw dataframe into an index into the
        # final data
        i = op.add(start_idx)

        def expected_dividend_adjustment(idx, symbol):
            sid = sids[symbol]
            return (
                1 -
                all_.ix[idx, ('ex_dividend', sid)] /
                all_.ix[idx - 1, ('close', sid)]
            )

        adjustments = [
            # ohlc
            {
                # dividends
                i(24): [Float64Multiply(
                    first_row=0,
                    last_row=i(24),
                    first_col=sids['AAPL'],
                    last_col=sids['AAPL'],
                    value=expected_dividend_adjustment(24, 'AAPL'),
                )],
                i(87): [Float64Multiply(
                    first_row=0,
                    last_row=i(87),
                    first_col=sids['AAPL'],
                    last_col=sids['AAPL'],
                    value=expected_dividend_adjustment(87, 'AAPL'),
                )],
                i(150): [Float64Multiply(
                    first_row=0,
                    last_row=i(150),
                    first_col=sids['AAPL'],
                    last_col=sids['AAPL'],
                    value=expected_dividend_adjustment(150, 'AAPL'),
                )],
                i(214): [Float64Multiply(
                    first_row=0,
                    last_row=i(214),
                    first_col=sids['AAPL'],
                    last_col=sids['AAPL'],
                    value=expected_dividend_adjustment(214, 'AAPL'),
                )],

                i(31): [Float64Multiply(
                    first_row=0,
                    last_row=i(31),
                    first_col=sids['MSFT'],
                    last_col=sids['MSFT'],
                    value=expected_dividend_adjustment(31, 'MSFT'),
                )],
                i(90): [Float64Multiply(
                    first_row=0,
                    last_row=i(90),
                    first_col=sids['MSFT'],
                    last_col=sids['MSFT'],
                    value=expected_dividend_adjustment(90, 'MSFT'),
                )],
                i(158): [Float64Multiply(
                    first_row=0,
                    last_row=i(158),
                    first_col=sids['MSFT'],
                    last_col=sids['MSFT'],
                    value=expected_dividend_adjustment(158, 'MSFT'),
                )],
                i(222): [Float64Multiply(
                    first_row=0,
                    last_row=i(222),
                    first_col=sids['MSFT'],
                    last_col=sids['MSFT'],
                    value=expected_dividend_adjustment(222, 'MSFT'),
                )],

                # splits
                i(108): [Float64Multiply(
                    first_row=0,
                    last_row=i(108),
                    first_col=sids['AAPL'],
                    last_col=sids['AAPL'],
                    value=1.0 / 7.0,
                )],
            },
        ] * (len(self.columns) - 1) + [
            # volume
            {
                i(108): [Float64Multiply(
                    first_row=0,
                    last_row=i(108),
                    first_col=sids['AAPL'],
                    last_col=sids['AAPL'],
                    value=7.0,
                )],
            }
        ]
        return pricing, adjustments

    def test_bundle(self):
        # # 耗时3秒以内
        ingest(TEST_BUNDLE_NAME)
        bundle = load(TEST_BUNDLE_NAME)
        sids = TEST_SIDS
        assert_equal(set(bundle.asset_finder.sids), set(sids))

        sessions = self.calendar.all_sessions
        actual = bundle.equity_daily_bar_reader.load_raw_arrays(
            self.columns,
            sessions[sessions.get_loc(self.start_date, 'bfill')],
            sessions[sessions.get_loc(self.end_date, 'ffill')],
            sids,
        )
        expected_pricing, expected_adjustments = self._expected_data(
            bundle.asset_finder,
        )
        assert_equal(actual, expected_pricing, array_decimal=2)

        adjustments_for_cols = bundle.adjustment_reader.load_adjustments(
            self.columns,
            sessions,
            pd.Index(sids),
        )

        for column, adjustments, expected in zip(self.columns,
                                                 adjustments_for_cols,
                                                 expected_adjustments):
            assert_equal(
                adjustments,
                expected,
                msg=column,
            )