from algorithm.base_algorithm import *
from queue import PriorityQueue

class Nova(BaseAlgorithm):
    def __init__(self, network: Network, tenants: List[Tenant]):
        super().__init__(network, tenants)

    def nova(self):
        forwarder_heap = PriorityQueue()
        for fwd in self.network.forwarder_set:
            forwarder_heap.put_nowait(fwd)

        for tenant in self.tenants:
            for instance in tenant.ins_list:
                best_fwd = forwarder_heap.get_nowait()
                best_fwd.cap_remain -= instance.cost
                instance.forwarder = best_fwd
                forwarder_heap.put_nowait(best_fwd)
