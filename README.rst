
.. image:: https://media.quantopian.com/logos/open_source/zipline-logo-03_.png
    :target: http://www.zipline.io
    :width: 212px
    :align: center
    :alt: Zipline

=============

Zipline是QUANTOPIAN开发的算法交易库。这是一个事件驱动，支持回测和实时交易的系统。
Zipline目前有一个免费的\ `回测平台 <https://www.quantopian.com>`__\ ，可以托管平台建立和执行交易策略。为直接使用A股数据进行回测，小幅度进行了改动（当前不支持分时回测）。


特别说明
========
- **只针对Windows10操作系统，参考环境**
- Windows10 64位
- python 3.6.3
- Visual Studio Community 2017


安装
====
    # 首先安装本频道的cswd odo blaze empyrical
    pip install git+https://github.com/liudengfeng/zipline

增加模块
========
- 从公共网站提取并更新A股数据（使用Windows任务计划程序） 
- 添加基础数据\ ``fundamentals``\ 模块 
- 增加\ ``bulitin``\ 模块
- 整合talib，增加\ ``quantalib``\ 模块

基础数据
========

-  股票日线交易数据
-  指数日线交易数据
-  融资融券
-  证监会行业、国证行业
-  财务数据及指标
-  股票概念

**注** 使用Windows计划任务管理，在指定时段自动采集更新数据

``Fundamentals``
================

``Fundamentals``\ 是一个容器类，类似\ ``Quantopian Fundamental Data``\ ，包含\ ``pipeline``\ 所需的基本数据。如资产负债表、利润表、现金流量表、财务指标及行业分类等等，暂不包含估值部分。其属性或是单个绑定列，或是数据集，以此生成\ ``pipeline``\ 中常用的自定义因子(\ ``Factor``)，过滤器(\ ``Filter``)或是分类器(\ ``Classifier``)。

``builtin``
===========

此模块包含常用的自定义因子、过滤器、分类器，以及通用的总体筛选函数。

``quantalib``
=============

整合适用于\ ``pipeline``\ 的\ ``talib``\ 。暂不包含模式识别。

回测案例
========

.. code:: ipython3

    %load_ext zipline

.. code:: ipython3

    %%zipline --start 2017-1-1 --end 2017-4-20 --capital-base 100000
    
    from six import viewkeys
    from zipline.api import (
        attach_pipeline,
        date_rules,
        order_target_percent,
        pipeline_output,
        record,
        schedule_function,
    )
    from zipline.finance import commission
    from zipline.pipeline import Pipeline
    from zipline.pipeline.factors import RSI
    
    
    def make_pipeline():
        rsi = RSI()
        return Pipeline(
            columns={
                'longs': rsi.top(3),
                'shorts': rsi.bottom(3),
            },
        )
    
    
    def rebalance(context, data):
    
        # Pipeline data will be a dataframe with boolean columns named 'longs' and
        # 'shorts'.
        pipeline_data = context.pipeline_data
        all_assets = pipeline_data.index
    
        longs = all_assets[pipeline_data.longs]
        shorts = all_assets[pipeline_data.shorts]
    
        record(universe_size=len(all_assets))
    
        # Build a 2x-leveraged, equal-weight, long-short portfolio.
        one_third = 1.0 / 3.0
        for asset in longs:
            order_target_percent(asset, one_third)
    
        for asset in shorts:
            order_target_percent(asset, -one_third)
    
        # Remove any assets that should no longer be in our portfolio.
        portfolio_assets = longs | shorts
        positions = context.portfolio.positions
        for asset in viewkeys(positions) - set(portfolio_assets):
            # This will fail if the asset was removed from our portfolio because it
            # was delisted.
            if data.can_trade(asset):
                order_target_percent(asset, 0)
    
    
    def initialize(context):
        attach_pipeline(make_pipeline(), 'my_pipeline')
    
        # Rebalance each day.  In daily mode, this is equivalent to putting
        # `rebalance` in our handle_data, but in minute mode, it's equivalent to
        # running at the start of the day each day.
        schedule_function(rebalance, date_rules.every_day())
    
        # Explicitly set the commission to the "old" value until we can
        # rebuild example data.
        # github.com/quantopian/zipline/blob/master/tests/resources/
        # rebuild_example_data#L105
        context.set_commission(commission.PerShare(cost=.0075, min_trade_cost=1.0))
    
    
    def before_trading_start(context, data):
        context.pipeline_data = pipeline_output('my_pipeline')


.. parsed-literal::

    [2017-12-09 20:29:33.920809] INFO: Loader: Read benchmark and treasury data for 000300 from 1990-10-31 to 2017-12-08
    [2017-12-09 20:29:49.959577] INFO: Performance: after split: asset: Equity(002836 [新宏泽]), amount: 1494.0, cost_basis: 30.03, last_sale_price: 62.300000000000004
    [2017-12-09 20:29:49.959577] INFO: Performance: returning cash: 0.0
    [2017-12-09 20:29:50.462507] INFO: Performance: after split: asset: Equity(300213 [佳讯飞鸿]), amount: -726.0, cost_basis: 11.61, last_sale_price: 22.830000000000002
    [2017-12-09 20:29:50.463506] INFO: Performance: returning cash: 0.0
    [2017-12-09 20:29:50.903947] INFO: Performance: after split: asset: Equity(000711 [京蓝科技]), amount: -402.0, cost_basis: 15.25, last_sale_price: 31.11
    [2017-12-09 20:29:50.903947] INFO: Performance: returning cash: 0.0
    [2017-12-09 20:29:52.262802] INFO: Performance: Simulated 71 trading days out of 71.
    [2017-12-09 20:29:52.262802] INFO: Performance: first open: 2017-01-03 01:31:00+00:00
    [2017-12-09 20:29:52.262802] INFO: Performance: last close: 2017-04-20 07:00:00+00:00
    

.. raw:: html


安装使用
========

-  `安装参考 <https://github.com/liudengfeng/BackTest/blob/master/zipline/docs/memo/1_install_zipline.md>`__
-  `自动刷新 <https://github.com/liudengfeng/BackTest/blob/master/zipline/docs/memo/2_auto_refresh.md>`__
-  `使用测试 <https://github.com/liudengfeng/BackTest/tree/master/zipline/docs/memo/pipeline>`__

后续
====

-  修正补充
-  进一步完善\ ``TensorBoard``
-  整合使用\ ``tensorflow``

交流
====

