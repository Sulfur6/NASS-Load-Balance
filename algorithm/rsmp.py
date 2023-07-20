import random
from algorithm.base_algorithm import *
from queue import PriorityQueue


class RSMP(BaseAlgorithm):
    def __init__(self, network: Network, tenants: List[Tenant]):
        super().__init__(network, tenants)

    def build_feasible_assignment(self, tenant):
        self.network.forwarder_set.sort()
        sharding = self.network.forwarder_set[:tenant.max_sharding_size]
        sharding.sort()
        sharding_heap = PriorityQueue()
        for fwd in sharding:
            sharding_heap.put_nowait(fwd)

        for ins in tenant.ins_list:
            best_fwd = sharding_heap.get_nowait()
            best_fwd.cap_remain -= ins.cost
            ins.forwarder = best_fwd
            sharding_heap.put_nowait(best_fwd)

    def main_procedure(self):
        for tenant in self.tenants:
            self.build_feasible_assignment(tenant)
