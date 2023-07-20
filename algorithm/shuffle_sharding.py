import itertools
import random
from algorithm.base_algorithm import *
from queue import PriorityQueue


class ShuffleSharding(BaseAlgorithm):
    def __init__(self, network: Network, tenants: List[Tenant]):
        super().__init__(network, tenants)

    def shuffle_sharding(self):
        max_sharding_size = -1
        for tenant in self.tenants:
            max_sharding_size = max(max_sharding_size, tenant.max_sharding_size)
        for tenant in self.tenants:
            sharding = random.sample(self.network.forwarder_set, max_sharding_size)
            tenant.ins_list.sort()
            # sharding.sort()
            # for i in range(tenant.ins_count):
            #     fwd = sharding[i % len(sharding)]
            #     ins = tenant.ins_list[i]
            #     fwd.cap_remain -= ins.cost
            #     ins.forwarder = fwd
            sharding_heap = PriorityQueue()
            for fwd in sharding:
                sharding_heap.put_nowait(fwd)

            for ins in tenant.ins_list:
                best_fwd = sharding_heap.get_nowait()
                best_fwd.cap_remain -= ins.cost
                ins.forwarder = best_fwd
                sharding_heap.put_nowait(best_fwd)
        # max_sharding_size = -1
        # for tenant in self.tenants:
        #     max_sharding_size = max(max_sharding_size, tenant.max_sharding_size)
        #
        # for tenant, sharding in zip(self.tenants,
        #                             itertools.combinations(self.network.forwarder_set, max_sharding_size)):
        #     formatted_sharding = random.sample(sharding, tenant.max_sharding_size)
        #     sharding_heap = PriorityQueue()
        #     for fwd in formatted_sharding:
        #         sharding_heap.put_nowait(fwd)
        #
        #     for ins in tenant.ins_list:
        #         best_fwd = sharding_heap.get_nowait()
        #         best_fwd.cap_remain -= ins.cost
        #         ins.forwarder = best_fwd
        #         sharding_heap.put_nowait(best_fwd)
