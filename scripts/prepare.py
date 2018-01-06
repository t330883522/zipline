"""
ingest
"""

import time
import zipline.data.bundles as bundles_module

import logbook
import sys

logbook.set_datetime_format('local')
logbook.StreamHandler(sys.stdout).push_application()
log = logbook.Logger('ingest'.upper())


def ingest_data(bundle_name):
    ingest_start_time = time.time()
    log.info('prepare "{}" dataset ......'.format(bundle_name))
    bundles_module.ingest(bundle_name, show_progress=True)
    duration = format(time.time() - ingest_start_time, '0.2f')
    log.info('duration {} seconds'.format(duration))


def main():
    from zipline.pipeline.fundamentals.ctable import convert_sql_data_to_bcolz
    ingest_start_time = time.time()
    convert_sql_data_to_bcolz()
    duration = format(time.time() - ingest_start_time, '0.2f')
    log.info('duration {} seconds'.format(duration))

    # keep last 2
    for bundle_name in ('cnstock','.test'):
        ingest_data(bundle_name)
        bundles_module.clean(bundle_name, keep_last=2)
    to_bcolz()

if __name__ == '__mani__':
    main()