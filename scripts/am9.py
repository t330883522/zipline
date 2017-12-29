"""
am 9 更新融资融券数据
"""

import time
import zipline.data.bundles as bundles_module

import logbook
import sys

logbook.set_datetime_format('local')
logbook.StreamHandler(sys.stdout).push_application()
log = logbook.Logger('ingest'.upper())


def to_bcolz():
    """转换部分sql数据到bcolz格式"""
    from zipline.pipeline.fundamentals.ctable import convert_sql_data_to_bcolz
    ingest_start_time = time.time()
    convert_sql_data_to_bcolz()
    duration = format(time.time() - ingest_start_time, '0.2f')
    log.info('耗时{}秒'.format(duration))