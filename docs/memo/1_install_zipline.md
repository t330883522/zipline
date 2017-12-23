
# 前置条件
+ 新建`zipline`环境（名称随意，以下均假定在此环境下安装）
+ 安装`cswd`数据包及改版后的`odo`及`blaze`

> pip install odo datashape blaze

> pip uninstall odo blaze

> pip install git+https://github.com/liudengfeng/cswd

> pip install git+https://github.com/liudengfeng/odo

> pip install git+https://github.com/liudengfeng/blaze

**注意**：如果升级`odo`和`blaze`包，请注意`networkx`用法更改部分

# 依赖包

## `pip`安装
+ `pip安装requirements.txt`所列包
+ 如有安装失败，请下载对应whl包
    + [下载whl网址](https://www.lfd.uci.edu/~gohlke/pythonlibs)
    + ![参考whl清单](https://github.com/liudengfeng/zipline/blob/master/docs/memo/images/whl_packages.PNG)

# 安装`zipline`
+ `clone`项目到本地
+ 使用模式
    + 进入`zipline`环境
    + 进入`setup.py`所在的目录
    + `python setup.py install`
+ 开发模式
    + 进入`zipline`环境
    + 进入`setup.py`所在的目录
    + `python setup.py build_ext --inplace`
    + `python setup.py develop`
