import random
from algorithm.base_algorithm import *


class RSMP(BaseAlgorithm):
    def __init__(self, network: Network, tenants: List[Tenant]):
        super().__init__(network, tenants)

    def build_feasible_assignment(self, tenant):
        self.network.forwarder_set.sort()
        sharding = self.network.forwarder_set[:tenant.ins_count]
        for i in range(tenant.ins_count):
            fwd = sharding[i]
            ins = tenant.ins_list[i]
            fwd.cap_remain -= ins.cost
            ins.forwarder = fwd

    def main_procedure(self):
        for tenant in self.tenants:
            self.build_feasible_assignment(tenant)
