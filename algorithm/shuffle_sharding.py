import random
from algorithm.base_algorithm import *

class ShuffleSharding(BaseAlgorithm):
    def __init__(self, network: Network, tenants: List[Tenant], sharding_size: int = 4):
        super().__init__(network, tenants)
        self.sharding_size = sharding_size

    def shuffle_sharding(self):
        for tenant in self.tenants:
            sharding = random.sample(self.network.forwarder_set, tenant.ins_count)
            sharding.sort()
            tenant.ins_list.sort()
            for i in range(tenant.ins_count):
                fwd = sharding[i]
                ins = tenant.ins_list[i]
                fwd.cap_remain -= ins.cost
                ins.forwarder = fwd
