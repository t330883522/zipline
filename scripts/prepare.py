"""
每周交易日后下午7点运行

每周六下午1点执行（网络数据在每周六上午完成数据再刷新）
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
    log.info('准备生成"{}"数据集（请稍候......)'.format(bundle_name))
    bundles_module.ingest(bundle_name, show_progress=True)
    duration = format(time.time() - ingest_start_time, '0.2f')
    log.info('耗时{}秒'.format(duration))


def to_bcolz():
    """转换部分sql数据到bcolz格式"""
    from zipline.pipeline.fundamentals.ctable import convert_sql_data_to_bcolz
    ingest_start_time = time.time()
    convert_sql_data_to_bcolz()
    duration = format(time.time() - ingest_start_time, '0.2f')
    log.info('耗时{}秒'.format(duration))

def main():
    # 同步刷新测试集
    # 保留最近2项数据包
    for bundle_name in ('cnstock','.test'):
        ingest_data(bundle_name)
        bundles_module.clean(bundle_name, keep_last=2)
    to_bcolz()