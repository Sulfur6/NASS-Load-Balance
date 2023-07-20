import itertools
from algorithm.base_algorithm import *
from queue import PriorityQueue


class SMNAAS(BaseAlgorithm):
    """
    Submodular based NAAS load balancing algorithm.
    With strong isolated constraint
    """

    def __init__(self, network: Network, tenants: List[Tenant]):
        super().__init__(network, tenants)

    def build_feasible_assignment(self, tenant):
        self.network.forwarder_set.sort()
        tenant.ins_list.sort()
        for sub_set in itertools.combinations(self.network.forwarder_set, tenant.max_sharding_size):
            valid_flag = True
            for other_set in self.assigned_set:
                if set(sub_set).issubset(other_set) or other_set.issubset(set(sub_set)):
                    valid_flag = False
                    break
            if valid_flag:
                sharding_heap = PriorityQueue()
                for fwd in sub_set:
                    sharding_heap.put_nowait(fwd)

                for ins in tenant.ins_list:
                    best_fwd = sharding_heap.get_nowait()
                    best_fwd.cap_remain -= ins.cost
                    ins.forwarder = best_fwd
                    sharding_heap.put_nowait(best_fwd)
                self.assigned_set.append(set(sub_set))
                break

    def main_procedure(self):
        for tenant in self.tenants:
            self.build_feasible_assignment(tenant)
