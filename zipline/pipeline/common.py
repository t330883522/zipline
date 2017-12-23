"""
Common constants for Pipeline.
"""
AD_FIELD_NAME = 'asof_date'
ANNOUNCEMENT_FIELD_NAME = 'announcement_date'
CASH_FIELD_NAME = 'cash'
DAYS_SINCE_PREV = 'days_since_prev'
DAYS_TO_NEXT = 'days_to_next'
FISCAL_QUARTER_FIELD_NAME = 'fiscal_quarter'
FISCAL_YEAR_FIELD_NAME = 'fiscal_year'
NEXT_ANNOUNCEMENT = 'next_announcement'
PREVIOUS_AMOUNT = 'previous_amount'
PREVIOUS_ANNOUNCEMENT = 'previous_announcement'

EVENT_DATE_FIELD_NAME = 'event_date'
SID_FIELD_NAME = 'sid'
TS_FIELD_NAME = 'timestamp'

BALANCE_SHEET_FIELDS = frozenset([SID_FIELD_NAME, AD_FIELD_NAME, TS_FIELD_NAME] + \
    ['A{}'.format(str(x).zfill(3)) for x in range(1,109)])

INCOME_STATEMENT_FIELDS = frozenset([SID_FIELD_NAME, AD_FIELD_NAME, TS_FIELD_NAME] + \
    ['B{}'.format(str(x).zfill(3)) for x in range(1,46)])

CASH_FLOW_STATEMENT_FIELDS = frozenset([SID_FIELD_NAME, AD_FIELD_NAME, TS_FIELD_NAME] + \
    ['C{}'.format(str(x).zfill(3)) for x in range(1,90)])

MAIN_RATIOS_FIELDS = frozenset([SID_FIELD_NAME, AD_FIELD_NAME, TS_FIELD_NAME] + \
    ['D{}'.format(str(x).zfill(3)) for x in range(1,20)])

EARNINGS_RATIOS_FIELDS = frozenset([SID_FIELD_NAME, AD_FIELD_NAME, TS_FIELD_NAME] + \
    ['E{}'.format(str(x).zfill(3)) for x in range(1,16)])

SOLVENCY_RATIOS_FIELDS = frozenset([SID_FIELD_NAME, AD_FIELD_NAME, TS_FIELD_NAME] + \
    ['F{}'.format(str(x).zfill(3)) for x in range(1,18)])

GROWTH_RATIOS_FIELDS = frozenset([SID_FIELD_NAME, AD_FIELD_NAME, TS_FIELD_NAME] + \
    ['G{}'.format(str(x).zfill(3)) for x in range(1,5)])

OPERATION_RATIOS_FIELDS = frozenset([SID_FIELD_NAME, AD_FIELD_NAME, TS_FIELD_NAME] + \
    ['H{}'.format(str(x).zfill(3)) for x in range(1,15)])
