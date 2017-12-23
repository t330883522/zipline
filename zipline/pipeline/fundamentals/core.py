import re
import pandas as pd

from zipline.utils.memoize import classlazyval
from ..data.dataset import BoundColumn

from .categories import (get_report_item_categories,
                         report_item_meta,
                         get_concept_categories)

from .normalize import (normalized_stock_region_data,
                        normalized_cninfo_industry_data,
                        normalized_csrc_industry_data,
                        normalized_special_treatment_data,
                        normalized_short_name_data,
                        normalized_time_to_market_data,
                        from_bcolz_data,
                        )

from ..common import (
    BALANCE_SHEET_FIELDS,
    INCOME_STATEMENT_FIELDS,
    CASH_FLOW_STATEMENT_FIELDS,
    MAIN_RATIOS_FIELDS,
    EARNINGS_RATIOS_FIELDS,
    SOLVENCY_RATIOS_FIELDS,
    GROWTH_RATIOS_FIELDS,
    OPERATION_RATIOS_FIELDS,
)

REPORT_ITEM_CODE_PATTERN = re.compile('[A-H]\d{3}')
CONCEPT_CODE_PATTERN = re.compile('C\d{6}')

_SUPER_SECTOR_NAMES = {
    1:'周期',
    2:'防御',
    3:'敏感',
}

_SECTOR_NAMES = {
    205: '可选消费', 
    206: '医疗保健', 
    207: '公用事业', 
    101: '基本材料', 
    102: '主要消费', 
    103: '金融服务', 
    104: '房地产', 
    308: '通讯服务', 
    309: '能源', 
    310: '工业领域', 
    311: '工程技术',
}


class Fundamentals(object):
    """股票基础数据集容器类"""
    @staticmethod
    def query_supper_sector_name(supper_sector_code):
        """查询超级部门行业名称（输入编码）"""
        return _SUPER_SECTOR_NAMES[supper_sector_code]

    @staticmethod
    def query_sector_name(sector_code):
        """查询部门行业名称（输入编码）"""
        return _SECTOR_NAMES[sector_code]

    @staticmethod
    def query_concept_name(field_code):
        """查询股票概念中文名称（输入字段编码）"""
        field_code = field_code.upper()
        if not re.match(CONCEPT_CODE_PATTERN, field_code):
            raise ValueError('概念编码由"C"与六位数字编码组合而成。如C021310')
        name_maps = get_concept_categories()
        return name_maps[field_code[1:]]

    @staticmethod
    def query_concept_code(name_key):
        """模糊查询概念编码（输入概念关键词）"""
        name_maps = get_concept_categories()
        df = pd.DataFrame([('C{}'.format(k), v) for k,v in name_maps.items()], 
                          columns=['code','name'])
        return df.loc[df.name.str.contains(name_key), ['code','name']]

    @staticmethod
    def query_report_item_name(item_code):
        """查询财务报告科目名称（输入科目编码）"""
        if not re.match(REPORT_ITEM_CODE_PATTERN, item_code):
            raise ValueError('财务科目编码由A-H大写字符和三位数字组合而成。如A001,H004')
        maps = get_report_item_categories()
        return maps[item_code]

    @staticmethod
    def query_report_item_code(name_key):
        """模糊查询财务科目编码（输入科目包含的关键字）"""
        return report_item_meta.loc[report_item_meta.name.str.contains(name_key), 
                                    ['code','name']]

    @staticmethod
    def has_column(column):
        """简单判定列是否存在于`Fundamentals`各数据集中"""
        return type(column) == BoundColumn

    @classlazyval
    def time_to_market(self):
        """上市日期（单列）"""
        return normalized_time_to_market_data().time_to_market

    @classlazyval
    def region(self):
        """股票所属地区（单列）"""
        return normalized_stock_region_data().name

    @classlazyval
    def treatment(self):
        """股票特别处理（单列）"""
        return normalized_special_treatment_data().treatment

    @classlazyval
    def short_name(self):
        """股票简称（单列）"""
        return normalized_short_name_data().short_name

    @classlazyval
    def csrc(self):
        """证监会行业数据集"""
        return normalized_csrc_industry_data()

    @classlazyval
    def cninfo(self):
        """巨潮行业数据集"""
        return normalized_cninfo_industry_data()

    @classlazyval
    def dividend(self):
        """年度股利（单列）"""
        return from_bcolz_data(table_name = 'adjustments').amount

    @classlazyval
    def concept(self):
        """股票概念数据集"""
        return from_bcolz_data(table_name = 'concept_stocks')

    @classlazyval
    def margin(self):
        """股票融资融券数据集"""
        return from_bcolz_data(table_name = 'margins')

    @classlazyval
    def balance_sheet(self):
        """资产负债表"""
        return from_bcolz_data(table_name = 'finance_reports', 
                               columns = BALANCE_SHEET_FIELDS,
                               expr_name = 'balance_sheet')

    @classlazyval
    def income_statement(self):
        """利润表"""
        return from_bcolz_data(table_name = 'finance_reports', 
                               columns = INCOME_STATEMENT_FIELDS,
                               expr_name = 'income_statement')

    @classlazyval
    def cash_flow(self):
        """现金流量表"""
        return from_bcolz_data(table_name = 'finance_reports', 
                               columns = CASH_FLOW_STATEMENT_FIELDS,
                               expr_name = 'cash_flow')

    @classlazyval
    def key_financial_indicators(self):
        """主要财务指标"""
        return from_bcolz_data(table_name = 'finance_reports', 
                               columns = MAIN_RATIOS_FIELDS,
                               expr_name = 'key_financial_indicators')

    @classlazyval
    def earnings_ratios(self):
        """盈利能力比率"""
        return from_bcolz_data(table_name = 'finance_reports', 
                               columns = EARNINGS_RATIOS_FIELDS,
                               expr_name = 'earnings_ratios')

    @classlazyval
    def solvency_ratios(self):
        """偿还能力比率"""
        return from_bcolz_data(table_name = 'finance_reports', 
                               columns = SOLVENCY_RATIOS_FIELDS,
                               expr_name = 'solvency_ratios')

    @classlazyval
    def growth_ratios(self):
        """成长能力比率"""
        return from_bcolz_data(table_name = 'finance_reports', 
                               columns = GROWTH_RATIOS_FIELDS,
                               expr_name = 'growth_ratios')

    @classlazyval
    def operation_ratios(self):
        """营运能力比率"""
        return from_bcolz_data(table_name = 'finance_reports', 
                               columns = OPERATION_RATIOS_FIELDS,
                               expr_name = 'operation_ratios')