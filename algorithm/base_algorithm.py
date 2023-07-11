from typing import List
from network.models import *
from tenant.models import *
import numpy as np


class BaseAlgorithm:
    """
    Base algorithm for all algorithm
    Implemented some function that could use in all algorithms
    """

    def __init__(self, network: Network, tenants: List[Tenant]):
        self.network = network
        self.tenants = tenants
        self.assigned_set = list()

    def MSE(self):
        capacity_usage = np.array([(fwd.capacity - fwd.cap_remain) for fwd in self.network.forwarder_set])
        return capacity_usage.var()

    def load_balance_factor(self):
        lbf_list = np.array([(fwd.capacity - fwd.cap_remain) / fwd.capacity for fwd in self.network.forwarder_set])
        return lbf_list.max()

    def max_influenced_count(self):
        if len(self.assigned_set) > 0:
            res = 0
            for i in range(len(self.tenants)):
                influenced_count = 0
                for j in range(len(self.tenants)):
                    if i == j:
                        continue
                    if self.assigned_set[j].issubset(self.assigned_set[i]):
                        influenced_count += 1
                res = max(res, influenced_count)
            return res

        res = 0
        for tenant in self.tenants:
            fwd_set = set()
            for ins in tenant.ins_list:
                if ins.forwarder:
                    fwd_set.add(ins.forwarder)
            self.assigned_set.append(fwd_set)

        for i in range(len(self.tenants)):
            influenced_count = 0
            for j in range(len(self.tenants)):
                if i == j:
                    continue
                if self.assigned_set[j].issubset(self.assigned_set[i]):
                    influenced_count += 1
            res = max(res, influenced_count)

        return res

    def unassigned_count(self):
        res = 0
        for tenant in self.tenants:
            for ins in tenant.ins_list:
                if not ins.forwarder:
                    res += 1
        return res
