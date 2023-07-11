import copy

from algorithm.nova import Nova
from algorithm.shuffle_sharding import ShuffleSharding
from algorithm.ljh import SMNAAS
from algorithm.rsmp import RSMP

from data_maker.data import NAASNetwork


def run_single_exp(fwd_count: int, fwd_cap_range: [int, int], tenant_count: int, ins_count_range: [int, int],
                   ins_cost_range: [int, int]):
    """
    运行单组实验。
    :param fwd_count: forwarder数量
    :param fwd_cap_range: forwarder容量的范围，list类型，上届下届
    :param tenant_count: 租户数量
    :param ins_count_range: 租户拥有的实例数量区间
    :param ins_cost_range: 租户拥有的实例的资源消耗区间
    :return: dict类型，其中MSE代表方差；LBF代表负载均衡因子，MIC代表最大可能受影响租户数，NC为未完成调度的实例数量，调试用。
    """
    NAASNetwork.genarate_network(fwd_count, fwd_cap_range)
    NAASNetwork.genarate_tenants(tenant_count, ins_count_range, ins_cost_range)

    nova = Nova(copy.deepcopy(NAASNetwork.network), copy.deepcopy(NAASNetwork.tenants))
    shuffle_sharding = ShuffleSharding(copy.deepcopy(NAASNetwork.network), copy.deepcopy(NAASNetwork.tenants))
    rsmp = RSMP(copy.deepcopy(NAASNetwork.network), copy.deepcopy(NAASNetwork.tenants))
    sm_naas = SMNAAS(copy.deepcopy(NAASNetwork.network), copy.deepcopy(NAASNetwork.tenants))

    nova.nova()
    shuffle_sharding.shuffle_sharding()
    rsmp.main_procedure()
    sm_naas.main_procedure()

    res = {
        "MSE": [nova.MSE(), shuffle_sharding.MSE(), rsmp.MSE(), sm_naas.MSE()],
        "LBF": [nova.load_balance_factor(), shuffle_sharding.load_balance_factor(), rsmp.load_balance_factor(),
                sm_naas.load_balance_factor()],
        "MIC": [nova.max_influenced_count(), shuffle_sharding.max_influenced_count(), rsmp.max_influenced_count(),
                sm_naas.max_influenced_count()],
        "NC": [nova.unassigned_count(), shuffle_sharding.unassigned_count(), rsmp.unassigned_count(),
               sm_naas.unassigned_count()],
    }
    return res
