import copy

from algorithm.nova import Nova
from algorithm.shuffle_sharding import ShuffleSharding
from algorithm.ljh import SMNAAS
from algorithm.rsmp import RSMP

from data_maker.data import NAASNetwork


def run_single_exp(fwd_count: int, fwd_cap: int, tenant_count: int, ins_cost_range: [int, int],
                   avg_ipc: int):
    """
    运行单组实验。
    :param fwd_count: forwarder数量
    :param fwd_cap_range: forwarder容量的范围，list类型，上届下届
    :param tenant_count: 租户数量
    :param ins_count_range: 租户拥有的实例数量区间
    :param ins_cost_range: 租户拥有的实例的资源消耗区间
    :return: dict类型，其中MSE代表方差；LBF代表负载均衡因子，MIC代表最大可能受影响租户数，NC为未完成调度的实例数量，调试用。
    """
    NAASNetwork.genarate_network(fwd_count, fwd_cap)
    NAASNetwork.genarate_tenants(tenant_count, ins_cost_range, avg_ipc)

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


def run_set_exp(fwd_count: int, fwd_cap: int, tc_range: [int, int], tc_step: int, ins_cost_range: [int, int],
                avg_ipc: int):
    res = {
        "MSE": {
            "NOVA": [],
            "SS": [],
            "RSMP": [],
            "NAAS": []
        },
        "LBF": {
            "NOVA": [],
            "SS": [],
            "RSMP": [],
            "NAAS": []
        },
        "MIC": {
            "NOVA": [],
            "SS": [],
            "RSMP": [],
            "NAAS": []
        },
        "NC": {
            "NOVA": [],
            "SS": [],
            "RSMP": [],
            "NAAS": []
        }
    }

    for tenant_count in range(tc_range[0], tc_range[1], tc_step):
        single_res = run_single_exp(fwd_count, fwd_cap, tenant_count, ins_cost_range, avg_ipc)
        for key in res.keys():
            res[key]["NOVA"].append(single_res[key][0])
            res[key]["SS"].append(single_res[key][1])
            res[key]["RSMP"].append(single_res[key][2])
            res[key]["NAAS"].append(single_res[key][3])
    return res


def run_set_exp_avgipc(fwd_count: int, fwd_cap: int, tenant_count, ins_cost_range: [int, int],
                       airange, ai_step):
    res = {
        "MSE": {
            "NOVA": [],
            "SS": [],
            "RSMP": [],
            "NAAS": []
        },
        "LBF": {
            "NOVA": [],
            "SS": [],
            "RSMP": [],
            "NAAS": []
        },
        "MIC": {
            "NOVA": [],
            "SS": [],
            "RSMP": [],
            "NAAS": []
        },
        "NC": {
            "NOVA": [],
            "SS": [],
            "RSMP": [],
            "NAAS": []
        }
    }
    for avg_ipc in range(airange[0], airange[1], ai_step):
        single_res = run_single_exp(fwd_count, fwd_cap, tenant_count, ins_cost_range, avg_ipc)
        for key in res.keys():
            res[key]["NOVA"].append(single_res[key][0])
            res[key]["SS"].append(single_res[key][1])
            res[key]["RSMP"].append(single_res[key][2])
            res[key]["NAAS"].append(single_res[key][3])
    return res


def repeat_MIC_exp(fwd_count: int, fwd_cap: int, tenant_count, ins_cost_range: [int, int], avg_ipc: int):
    pre_sum_mic = {
        "NOVA": [],
        "SS": [],
        "RSMP": [],
        "NAAS": []
    }
    cur_stat = {
        "NOVA": 0,
        "SS": 0,
        "RSMP": 0,
        "NAAS": 0
    }
    final_count = {
        "NOVA": {},
        "SS": {},
        "RSMP": {},
        "NAAS": {}
    }
    name_mapping = {
        "NOVA": 0,
        "SS": 1,
        "RSMP": 2,
        "NAAS": 3
    }
    for i in range(1000):
        single_res = run_single_exp(fwd_count, fwd_cap, tenant_count, ins_cost_range, avg_ipc)
        for key in name_mapping.keys():
            count = single_res["MIC"][name_mapping[key]]
            cur_stat[key] += count
            if count not in final_count[key]:
                final_count[key][count] = 0
            final_count[key][count] += 1
        if (i + 1) % 100 == 0:
            for key in name_mapping.keys():
                pre_sum_mic[key].append(cur_stat[key])
    return pre_sum_mic, final_count
