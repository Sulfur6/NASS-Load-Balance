import copy

from algorithm.nova import Nova
from algorithm.shuffle_sharding import ShuffleSharding
from algorithm.ljh import SMNAAS
from algorithm.rsmp import RSMP

from data_maker.data import NAASNetwork


def run_single_exp(fwd_count, fwd_cap_range, tenant_count, ins_count_range, ins_cost_range):
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
