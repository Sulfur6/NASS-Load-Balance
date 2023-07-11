# 项目简介
NASS Load Balance 实验代码

# 项目依赖
* Python >= 3.5

# 环境配置
## 创建虚拟环境

```
virtualenv venv
```

## 激活虚拟环境

```
source venv/bin/activate
```

如果用pycharm的话虚拟环境可以一键创建

## 安装依赖

```
pip install -r requirements.txt
```

# 项目结构简介
algorithm 模块实现了所有的算法，包括算法基类及四种对比算法。

network与tenant模块实现了网络中的各种角色。

data_maker模块用来生成数据，主要通过random.randint随机生成。

experiment中实现与实验相关的代码，目前只实现了一个run_single_exp函数:
```angular2html
运行单组实验。
:param fwd_count: forwarder数量
:param fwd_cap_range: forwarder容量的范围，list类型，上届下届
:param tenant_count: 租户数量
:param ins_count_range: 租户拥有的实例数量区间
:param ins_cost_range: 租户拥有的实例的资源消耗区间
:return: dict类型，其中MSE代表方差；LBF代表负载均衡因子，MIC代表最大可能受影响租户数，NC为未完成调度的实例数量，调试用。
```