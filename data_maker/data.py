from network.models import *
from tenant.models import *
import random

class NAASNetwork:
    network: Network = None
    tenants: List[Tenant] = None

    @classmethod
    def genarate_network(cls, fwd_count: int, fwd_cap_range: [int, int]) -> None:
        fwd_set: List[Forwarder] = []
        capacity = random.randint(fwd_cap_range[0], fwd_cap_range[1])
        for i in range(fwd_count):
            fwd_set.append(Forwarder(i, capacity))
        cls.network = Network(fwd_set)

    @classmethod
    def genarate_tenants(cls, tenant_count: int, ins_count_range: [int, int], ins_cost_range: [int, int]) -> None:
        cls.tenants: List[Tenant] = []
        for i in range(tenant_count):
            ins_count = random.randint(ins_count_range[0], ins_count_range[1])
            ins_list: List[Instance] = []
            for j in range(ins_count):
                ins_cost = random.randint(ins_cost_range[0], ins_count_range[1])
                ins_list.append(Instance(j, ins_cost))
            cls.tenants.append(Tenant(ins_list))
