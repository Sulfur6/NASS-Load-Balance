from network.models import *
from tenant.models import *
import random


def generate_avg_based_list(length, avg, a, b) -> List[int]:
    """
    :param length: length of the list
    :param avg: pre-designed average value
    :param a: lower bound of value in the list
    :param b: upper bound of value in the list
    :return: a list contains integers with average == avg and elements ranges from a to b
    """
    generated_list: List[int] = []
    for i in range(length // 2):
        value = random.randint(a, b)
        next_value = 2 * avg - value
        generated_list.append(value)
        generated_list.append(next_value)
    if length % 2 != 0:
        generated_list.append(avg)

    # factor = (avg - 2) / 38
    # _s_len = int(length * factor)
    # _s_list = [40 for i in range(_s_len)]
    # _l_len = length - _s_len
    # _l_list = [2 for i in range(_l_len)]
    # generated_list = _s_list + _l_list
    # random.shuffle(generated_list)
    return generated_list


class NAASNetwork:
    """
    class used to build a naas network
    """
    network: Network = None
    tenants: List[Tenant] = None

    @classmethod
    def genarate_network(cls, fwd_count: int, fwd_cap: int) -> None:
        fwd_set: List[Forwarder] = []
        capacity = fwd_cap
        for i in range(fwd_count):
            fwd_set.append(Forwarder(i, capacity))
        cls.network = Network(fwd_set)

    @classmethod
    def genarate_tenants(cls, tenant_count: int, ins_cost_range: [int, int],
                         avg_ipc: int) -> None:
        k_list = generate_avg_based_list(tenant_count, avg_ipc, avg_ipc - 2, avg_ipc + 2)
        cls.tenants: List[Tenant] = []
        for i in range(tenant_count):
            ins_count = random.randint(k_list[i] * 2, k_list[i] * 3)
            ins_list: List[Instance] = []
            for j in range(ins_count):
                ins_cost = random.randint(ins_cost_range[0], ins_cost_range[1])
                ins_list.append(Instance(j, ins_cost))
            cls.tenants.append(Tenant(ins_list, k_list[i]))
