"""
am 9 margin data to bcolz
"""

import time
import zipline.data.bundles as bundles_module

import logbook
import sys

logbook.set_datetime_format('local')
logbook.StreamHandler(sys.stdout).push_application()
log = logbook.Logger('ingest'.upper())


def main():
    """margin data to bcolz"""
    from zipline.pipeline.fundamentals.ctable import convert_sql_data_to_bcolz
    ingest_start_time = time.time()
    convert_sql_data_to_bcolz('margins')
    duration = format(time.time() - ingest_start_time, '0.2f')
    log.info('duration:{} seconds'.format(duration))